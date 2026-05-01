import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, normal_init
from torch.nn.modules.batchnorm import _BatchNorm

from core.utils.layer_utils import get_norm
from .resnet_backbone import resnet_spec


class HTDecoderLayer(nn.Module):
    """Hybrid Transformer decoder layer for geometry feature enhancement."""

    def __init__(self, d_model=256, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, need_weights=False)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        mem = tgt if memory is None else memory
        tgt2, _ = self.cross_attn(tgt, mem, mem, need_weights=False)
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class HTDecoder(nn.Module):
    def __init__(self, d_model=256, num_layers=1, nhead=8, dim_feedforward=512):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                HTDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):
        out = tgt
        for layer in self.layers:
            out = layer(out, memory)
        return self.norm(out)


class RotWithRegionHead(nn.Module):
    def __init__(
        self,
        cfg,
        in_channels,
        num_layers=3,
        num_filters=256,
        kernel_size=3,
        output_kernel_size=1,
        rot_output_dim=3,
        mask_output_dim=1,
        freeze=False,
        num_classes=1,
        rot_class_aware=False,
        mask_class_aware=False,
        region_class_aware=False,
        num_regions=8,
        norm="BN",
        num_gn_groups=32,
        memory_channels=None,
    ):
        super().__init__()

        self.freeze = freeze
        self.concat = cfg.MODEL.CDPN.ROT_HEAD.ROT_CONCAT if hasattr(cfg.MODEL.CDPN.ROT_HEAD, "ROT_CONCAT") else False
        self.num_regions = num_regions
        self.num_filters = num_filters

        self.rot_output_dim = rot_output_dim * num_classes if rot_class_aware else rot_output_dim
        self.mask_output_dim = mask_output_dim * num_classes if mask_class_aware else mask_output_dim
        self.region_output_dim = (num_regions + 1) * num_classes if region_class_aware else (num_regions + 1)

        self.use_htd = bool(getattr(cfg.MODEL.CDPN.ROT_HEAD, "USE_HTD", False))
        htd_layers = int(getattr(cfg.MODEL.CDPN.ROT_HEAD, "HTD_NUM_LAYERS", 1))
        htd_nhead = int(getattr(cfg.MODEL.CDPN.ROT_HEAD, "HTD_NHEAD", 8))
        htd_ff = int(getattr(cfg.MODEL.CDPN.ROT_HEAD, "HTD_FF", 512))

        assert kernel_size in (2, 3, 4), "Only support kernel 2, 3, and 4"
        assert output_kernel_size in (1, 3), "Only support output kernel 1 and 3"
        assert num_regions > 1, f"Only support num_regions > 1, but got {num_regions}"

        padding = 1 if kernel_size == 3 else (0 if kernel_size == 2 else 1)
        output_padding = 1 if kernel_size == 3 else 0
        pad = 1 if output_kernel_size == 3 else 0

        self.features = nn.ModuleList()
        if self.concat:
            _, _, channels, _ = resnet_spec[cfg.MODEL.CDPN.BACKBONE.NUM_LAYERS]
            self.features.append(
                nn.ConvTranspose2d(
                    in_channels,
                    num_filters,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
            self.features.append(nn.ReLU(inplace=True))
            for i in range(num_layers):
                self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
                self.features.append(
                    nn.Conv2d(
                        num_filters + channels[-2 - i],
                        num_filters,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    )
                )
                self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
                self.features.append(nn.ReLU(inplace=True))
                self.features.append(
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
                )
                self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
                self.features.append(nn.ReLU(inplace=True))
        else:
            self.features.append(
                nn.ConvTranspose2d(
                    in_channels,
                    num_filters,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
            self.features.append(nn.ReLU(inplace=True))
            for i in range(num_layers):
                if i >= 1:
                    self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
                self.features.append(
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
                )
                self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
                self.features.append(nn.ReLU(inplace=True))
                self.features.append(
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
                )
                self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
                self.features.append(nn.ReLU(inplace=True))

        self.memory_channels = tuple(int(ch) for ch in (memory_channels or []))
        self.memory_projs = nn.ModuleDict()
        for ch in self.memory_channels:
            self.memory_projs[str(ch)] = nn.Conv2d(ch, num_filters, kernel_size=1, bias=False)
        self.memory_scale_logits = (
            nn.Parameter(torch.zeros(len(self.memory_channels))) if self.memory_channels else None
        )
        self.last_htd_memory_shape = None

        if self.use_htd:
            self.htd = HTDecoder(
                d_model=num_filters,
                num_layers=htd_layers,
                nhead=htd_nhead,
                dim_feedforward=htd_ff,
            )

        self.final_conv = nn.Conv2d(
            num_filters,
            self.mask_output_dim + self.rot_output_dim + self.region_output_dim,
            kernel_size=output_kernel_size,
            padding=pad,
            bias=True,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def _project_memory(self, memory_feature, target_channels):
        if memory_feature is None:
            return None
        _, memory_channels, _, _ = memory_feature.shape
        if memory_channels == target_channels:
            return memory_feature

        key = str(memory_channels)
        if key not in self.memory_projs:
            expected = ", ".join(self.memory_projs.keys()) or "none"
            raise ValueError(
                f"Unexpected HTD memory channels {memory_channels}; registered channels: {expected}"
            )
        return self.memory_projs[key](memory_feature)

    def _build_memory_map(self, memory_features, target_shape):
        if memory_features is None:
            return None
        if isinstance(memory_features, torch.Tensor):
            memory_features = [memory_features]

        _, target_channels, target_h, target_w = target_shape
        projected_features = []
        logits = []
        for idx, memory_feature in enumerate(memory_features):
            if memory_feature is None:
                continue
            projected = self._project_memory(memory_feature, target_channels)
            if projected.shape[-2:] != (target_h, target_w):
                projected = F.interpolate(projected, size=(target_h, target_w), mode="bilinear", align_corners=False)
            projected_features.append(projected)
            if self.memory_scale_logits is not None and idx < len(self.memory_scale_logits):
                logits.append(self.memory_scale_logits[idx])

        if not projected_features:
            return None

        if len(projected_features) == 1:
            return projected_features[0]

        stacked = torch.stack(projected_features, dim=0)
        if len(logits) == len(projected_features):
            weights = torch.softmax(torch.stack(logits), dim=0).view(-1, 1, 1, 1, 1)
        else:
            weights = stacked.new_full((len(projected_features), 1, 1, 1, 1), 1.0 / len(projected_features))
        return (stacked * weights).sum(dim=0)

    def _apply_htd_if_enabled(self, x, memory_features=None):
        if not self.use_htd:
            self.last_htd_memory_shape = None
            return x

        b, c, h, w = x.shape
        tgt = x.flatten(2).permute(0, 2, 1).contiguous()
        memory_map = self._build_memory_map(memory_features, x.shape)
        if memory_map is None:
            mem = None
            self.last_htd_memory_shape = None
        else:
            mem = memory_map.flatten(2).permute(0, 2, 1).contiguous()
            self.last_htd_memory_shape = tuple(memory_map.shape)

        out = self.htd(tgt, mem)
        return out.permute(0, 2, 1).contiguous().view(b, c, h, w)

    def _split_outputs(self, x):
        mask = x[:, : self.mask_output_dim, :, :]
        xyz = x[:, self.mask_output_dim : self.mask_output_dim + self.rot_output_dim, :, :]
        region = x[:, self.mask_output_dim + self.rot_output_dim :, :, :]

        bs, _, h, w = xyz.shape
        xyz = xyz.view(bs, 3, self.rot_output_dim // 3, h, w)
        coor_x = xyz[:, 0, :, :, :]
        coor_y = xyz[:, 1, :, :, :]
        coor_z = xyz[:, 2, :, :, :]
        return mask, coor_x, coor_y, coor_z, region

    def forward(self, x, x_f64=None, x_f32=None, x_f16=None):
        if self.concat:
            for i, layer in enumerate(self.features):
                if i == 3:
                    x = layer(torch.cat([x, x_f16], 1))
                elif i == 12:
                    x = layer(torch.cat([x, x_f32], 1))
                elif i == 21:
                    x = layer(torch.cat([x, x_f64], 1))
                else:
                    x = layer(x)
        else:
            for layer in self.features:
                x = layer(x)

        memory_features = [x_f64, x_f32, x_f16]
        x = self._apply_htd_if_enabled(x, memory_features)
        x = self.final_conv(x)
        outputs = self._split_outputs(x)

        if self.freeze:
            return tuple(output.detach() for output in outputs)
        return outputs
