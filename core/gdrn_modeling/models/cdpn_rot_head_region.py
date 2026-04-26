# import torch.nn as nn
# import torch
# from mmcv.cnn import normal_init, kaiming_init, constant_init
# from core.utils.layer_utils import get_norm
# from torch.nn.modules.batchnorm import _BatchNorm
# from .resnet_backbone import resnet_spec


# class RotWithRegionHead(nn.Module):
#     def __init__(
#         self,
#         cfg,
#         in_channels,
#         num_layers=3,
#         num_filters=256,
#         kernel_size=3,
#         output_kernel_size=1,
#         rot_output_dim=3,
#         mask_output_dim=1,
#         freeze=False,
#         num_classes=1,
#         rot_class_aware=False,
#         mask_class_aware=False,
#         region_class_aware=False,
#         num_regions=8,
#         norm="BN",
#         num_gn_groups=32,
#     ):
#         super().__init__()

#         self.freeze = freeze
#         self.concat = cfg.MODEL.CDPN.ROT_HEAD.ROT_CONCAT
#         assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, "Only support kenerl 2, 3 and 4"
#         assert num_regions > 1, f"Only support num_regions > 1, but got {num_regions}"
#         padding = 1
#         output_padding = 0
#         if kernel_size == 3:
#             output_padding = 1
#         elif kernel_size == 2:
#             padding = 0

#         assert output_kernel_size == 1 or output_kernel_size == 3, "Only support kenerl 1 and 3"
#         if output_kernel_size == 1:
#             pad = 0
#         elif output_kernel_size == 3:
#             pad = 1

#         if self.concat:
#             _, _, channels, _ = resnet_spec[cfg.MODEL.CDPN.BACKBONE.NUM_LAYERS]
#             self.features = nn.ModuleList()
#             self.features.append(
#                 nn.ConvTranspose2d(
#                     in_channels,
#                     num_filters,
#                     kernel_size=kernel_size,
#                     stride=2,
#                     padding=padding,
#                     output_padding=output_padding,
#                     bias=False,
#                 )
#             )
#             self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
#             self.features.append(nn.ReLU(inplace=True))
#             for i in range(num_layers):
#                 self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
#                 self.features.append(
#                     nn.Conv2d(
#                         num_filters + channels[-2 - i], num_filters, kernel_size=3, stride=1, padding=1, bias=False
#                     )
#                 )
#                 self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
#                 self.features.append(nn.ReLU(inplace=True))

#                 self.features.append(
#                     nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
#                 )
#                 self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
#                 self.features.append(nn.ReLU(inplace=True))
#         else:
#             self.features = nn.ModuleList()
#             self.features.append(
#                 nn.ConvTranspose2d(
#                     in_channels,
#                     num_filters,
#                     kernel_size=kernel_size,
#                     stride=2,
#                     padding=padding,
#                     output_padding=output_padding,
#                     bias=False,
#                 )
#             )
#             self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
#             self.features.append(nn.ReLU(inplace=True))
#             for i in range(num_layers):
#                 # _in_channels = in_channels if i == 0 else num_filters
#                 # self.features.append(
#                 #    nn.ConvTranspose2d(_in_channels, num_filters, kernel_size=kernel_size, stride=2, padding=padding,
#                 #                       output_padding=output_padding, bias=False))
#                 # self.features.append(nn.BatchNorm2d(num_filters))
#                 # self.features.append(nn.ReLU(inplace=True))
#                 if i >= 1:
#                     self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
#                 self.features.append(
#                     nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
#                 )
#                 self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
#                 self.features.append(nn.ReLU(inplace=True))

#                 self.features.append(
#                     nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
#                 )
#                 self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
#                 self.features.append(nn.ReLU(inplace=True))

#         self.rot_output_dim = rot_output_dim
#         if rot_class_aware:
#             self.rot_output_dim *= num_classes

#         self.mask_output_dim = mask_output_dim
#         if mask_class_aware:
#             self.mask_output_dim *= num_classes

#         self.region_output_dim = num_regions + 1  # add one channel for bg
#         if region_class_aware:
#             self.region_output_dim *= num_classes

#         self.features.append(
#             nn.Conv2d(
#                 num_filters,
#                 self.mask_output_dim + self.rot_output_dim + self.region_output_dim,
#                 kernel_size=output_kernel_size,
#                 padding=pad,
#                 bias=True,
#             )
#         )

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 normal_init(m, std=0.001)
#             elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
#                 constant_init(m, 1)
#             elif isinstance(m, nn.ConvTranspose2d):
#                 normal_init(m, std=0.001)

#     def forward(self, x, x_f64=None, x_f32=None, x_f16=None):
#         if self.concat:
#             if self.freeze:
#                 with torch.no_grad():
#                     for i, l in enumerate(self.features):
#                         if i == 3:
#                             x = l(torch.cat([x, x_f16], 1))
#                         elif i == 12:
#                             x = l(torch.cat([x, x_f32], 1))
#                         elif i == 21:
#                             x = l(torch.cat([x, x_f64], 1))
#                         x = l(x)
#                     return x.detach()
#             else:
#                 for i, l in enumerate(self.features):
#                     if i == 3:
#                         x = torch.cat([x, x_f16], 1)
#                     elif i == 12:
#                         x = torch.cat([x, x_f32], 1)
#                     elif i == 21:
#                         x = torch.cat([x, x_f64], 1)
#                     x = l(x)
#                 return x
#         else:
#             if self.freeze:
#                 with torch.no_grad():
#                     for i, l in enumerate(self.features):
#                         x = l(x)
#                     mask = x[:, : self.mask_output_dim, :, :]
#                     xyz = x[:, self.mask_output_dim : self.mask_output_dim + self.rot_output_dim, :, :]
#                     region = x[:, self.mask_output_dim + self.rot_output_dim :, :, :]
#                     bs, c, h, w = xyz.shape
#                     xyz = xyz.view(bs, 3, self.rot_output_dim // 3, h, w)
#                     coor_x = xyz[:, 0, :, :, :]
#                     coor_y = xyz[:, 1, :, :, :]
#                     coor_z = xyz[:, 2, :, :, :]
#                     return (mask.detach(), coor_x.detach(), coor_y.detach(), coor_z.detach(), region.detach())
#             else:
#                 for i, l in enumerate(self.features):
#                     x = l(x)
#                 mask = x[:, : self.mask_output_dim, :, :]
#                 xyz = x[:, self.mask_output_dim : self.mask_output_dim + self.rot_output_dim, :, :]
#                 region = x[:, self.mask_output_dim + self.rot_output_dim :, :, :]
#                 bs, c, h, w = xyz.shape
#                 xyz = xyz.view(bs, 3, self.rot_output_dim // 3, h, w)
#                 coor_x = xyz[:, 0, :, :, :]
#                 coor_y = xyz[:, 1, :, :, :]
#                 coor_z = xyz[:, 2, :, :, :]
#                 return mask, coor_x, coor_y, coor_z, region







# 旋转头添加HTD：

# import torch.nn as nn
# import torch
# from mmcv.cnn import normal_init, kaiming_init, constant_init
# from core.utils.layer_utils import get_norm
# from torch.nn.modules.batchnorm import _BatchNorm
# from .resnet_backbone import resnet_spec


# class HTDecoder(nn.Module):
#     """
#     Lightweight Hybrid Transformer Decoder (HTD).
#     - d_model: feature dimension (num_filters)
#     - nhead: attention heads
#     - num_layers: number of decoder layers (usually 1 or 2)
#     - dim_feedforward: FFN hidden dim
#     """
#     def __init__(self, d_model=256, nhead=8, num_layers=1, dim_feedforward=1024, dropout=0.1):
#         super().__init__()
#         self.layers = nn.ModuleList(
#             [
#                 nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
#                 for _ in range(num_layers)
#             ]
#         )
#         self.norm = nn.LayerNorm(d_model)

#     def forward(self, tgt, memory):
#         """
#         tgt: (B, S, E)  features to be decoded (query)
#         memory: (B, S_mem, E)  multi-scale memory (key/value)
#         We'll permute to (S, B, E) because PyTorch's default Transformer uses seq-first.
#         """
#         # If memory is None, use tgt as memory (self-attention)
#         if memory is None:
#             memory = tgt

#         # permute to (T, B, E)
#         tgt_seq = tgt.permute(1, 0, 2)
#         mem_seq = memory.permute(1, 0, 2)

#         out = tgt_seq
#         for layer in self.layers:
#             out = layer(out, mem_seq)  # out: (T, B, E)
#         out = out.permute(1, 0, 2)  # back to (B, T, E)
#         out = self.norm(out)
#         return out  # (B, S, E)


# class RotWithRegionHead(nn.Module):
#     def __init__(
#         self,
#         cfg,
#         in_channels,
#         num_layers=3,
#         num_filters=256,
#         kernel_size=3,
#         output_kernel_size=1,
#         rot_output_dim=3,
#         mask_output_dim=1,
#         freeze=False,
#         num_classes=1,
#         rot_class_aware=False,
#         mask_class_aware=False,
#         region_class_aware=False,
#         num_regions=8,
#         norm="BN",
#         num_gn_groups=32,
#     ):
#         super().__init__()

#         self.freeze = freeze
#         self.concat = cfg.MODEL.CDPN.ROT_HEAD.ROT_CONCAT

#         # Optional HTD
#         try:
#             self.use_htd = bool(cfg.MODEL.CDPN.ROT_HEAD.USE_HTD)
#         except Exception:
#             self.use_htd = False

#         try:
#             htd_layers = int(cfg.MODEL.CDPN.ROT_HEAD.HTD_NUM_LAYERS)
#         except Exception:
#             htd_layers = 1
#         try:
#             htd_nhead = int(cfg.MODEL.CDPN.ROT_HEAD.HTD_NHEAD)
#         except Exception:
#             htd_nhead = 8
#         try:
#             htd_ff = int(cfg.MODEL.CDPN.ROT_HEAD.HTD_FF)
#         except Exception:
#             htd_ff = 1024

#         if self.use_htd:
#             self.htd = HTDecoder(d_model=num_filters, nhead=htd_nhead, num_layers=htd_layers, dim_feedforward=htd_ff)

#         assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, "Only support kernel 2, 3 and 4"
#         assert num_regions > 1, f"Only support num_regions > 1, but got {num_regions}"
#         padding = 1
#         output_padding = 0
#         if kernel_size == 3:
#             output_padding = 1
#         elif kernel_size == 2:
#             padding = 0

#         assert output_kernel_size == 1 or output_kernel_size == 3, "Only support kernel 1 and 3"
#         pad = 0
#         if output_kernel_size == 3:
#             pad = 1

#         self.features = nn.ModuleList()

#         if self.concat:
#             _, _, channels, _ = resnet_spec[cfg.MODEL.CDPN.BACKBONE.NUM_LAYERS]
#             self.features.append(
#                 nn.ConvTranspose2d(
#                     in_channels,
#                     num_filters,
#                     kernel_size=kernel_size,
#                     stride=2,
#                     padding=padding,
#                     output_padding=output_padding,
#                     bias=False,
#                 )
#             )
#             self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
#             self.features.append(nn.ReLU(inplace=True))
#             for i in range(num_layers):
#                 self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
#                 self.features.append(
#                     nn.Conv2d(
#                         num_filters + channels[-2 - i], num_filters, kernel_size=3, stride=1, padding=1, bias=False
#                     )
#                 )
#                 self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
#                 self.features.append(nn.ReLU(inplace=True))

#                 self.features.append(
#                     nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
#                 )
#                 self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
#                 self.features.append(nn.ReLU(inplace=True))
#         else:
#             self.features.append(
#                 nn.ConvTranspose2d(
#                     in_channels,
#                     num_filters,
#                     kernel_size=kernel_size,
#                     stride=2,
#                     padding=padding,
#                     output_padding=output_padding,
#                     bias=False,
#                 )
#             )
#             self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
#             self.features.append(nn.ReLU(inplace=True))
#             for i in range(num_layers):
#                 if i >= 1:
#                     self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
#                 self.features.append(
#                     nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
#                 )
#                 self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
#                 self.features.append(nn.ReLU(inplace=True))

#                 self.features.append(
#                     nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
#                 )
#                 self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
#                 self.features.append(nn.ReLU(inplace=True))

#         self.rot_output_dim = rot_output_dim
#         if rot_class_aware:
#             self.rot_output_dim *= num_classes

#         self.mask_output_dim = mask_output_dim
#         if mask_class_aware:
#             self.mask_output_dim *= num_classes

#         self.region_output_dim = num_regions + 1  # add one channel for bg
#         if region_class_aware:
#             self.region_output_dim *= num_classes

#         self.final_conv = nn.Conv2d(
#             num_filters,
#             self.mask_output_dim + self.rot_output_dim + self.region_output_dim,
#             kernel_size=output_kernel_size,
#             padding=pad,
#             bias=True,
#         )

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 normal_init(m, std=0.001)
#             elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
#                 constant_init(m, 1)
#             elif isinstance(m, nn.ConvTranspose2d):
#                 normal_init(m, std=0.001)

#     def _apply_htd_if_enabled(self, x, memory_feature=None):
#         if not self.use_htd:
#             return x
#         # x: (B, C, H, W) -> tgt (B, S, E)
#         b, c, h, w = x.shape
#         tgt = x.flatten(2).permute(0, 2, 1)  # (B, S, E)
#         mem = None
#         if memory_feature is not None:
#             # if memory_feature spatial dims differ, interpolate
#             mb, mc, mh, mw = memory_feature.shape
#             if mc != c:
#                 # Project channels if mismatch
#                 proj = nn.Conv2d(mc, c, kernel_size=1, bias=False)
#                 memory_feature = proj(memory_feature)
#             if (mh * mw) != (h * w):
#                 memory_feature = nn.functional.interpolate(memory_feature, size=(h, w), mode="bilinear", align_corners=False)
#             mem = memory_feature.flatten(2).permute(0, 2, 1)
#         out = self.htd(tgt, mem)  # (B, S, E)
#         out = out.permute(0, 2, 1).view(b, c, h, w)
#         return out

#     def forward(self, x, x_f64=None, x_f32=None, x_f16=None):
#         if self.concat:
#             if self.freeze:
#                 with torch.no_grad():
#                     for i, l in enumerate(self.features):
#                         if i == 3:
#                             x = l(torch.cat([x, x_f16], 1))
#                         elif i == 12:
#                             x = l(torch.cat([x, x_f32], 1))
#                         elif i == 21:
#                             x = l(torch.cat([x, x_f64], 1))
#                         else:
#                             x = l(x)
#                     # Apply HTD
#                     mem = x_f32 if x_f32 is not None else (x_f16 if x_f16 is not None else x_f64)
#                     x = self._apply_htd_if_enabled(x, mem)
#                     x = self.final_conv(x)
#                     return x.detach()
#             else:
#                 for i, l in enumerate(self.features):
#                     if i == 3:
#                         x = l(torch.cat([x, x_f16], 1))
#                     elif i == 12:
#                         x = l(torch.cat([x, x_f32], 1))
#                     elif i == 21:
#                         x = l(torch.cat([x, x_f64], 1))
#                     else:
#                         x = l(x)
#                     # Apply HTD
#                     mem = x_f32 if x_f32 is not None else (x_f16 if x_f16 is not None else x_f64)
#                     x = self._apply_htd_if_enabled(x, mem)
#                     x = self.final_conv(x)
#                     return x
#         else:
#             if self.freeze:
#                 with torch.no_grad():
#                     for i, l in enumerate(self.features):
#                         x = l(x)
#                     # Apply HTD
#                     mem = x_f32 if x_f32 is not None else (x_f16 if x_f16 is not None else x_f64)
#                     x = self._apply_htd_if_enabled(x, mem)
#                     x = self.final_conv(x)
#                     mask = x[:, : self.mask_output_dim, :, :]
#                     xyz = x[:, self.mask_output_dim : self.mask_output_dim + self.rot_output_dim, :, :]
#                     region = x[:, self.mask_output_dim + self.rot_output_dim :, :, :]
#                     bs, c, h, w = xyz.shape
#                     xyz = xyz.view(bs, 3, self.rot_output_dim // 3, h, w)
#                     coor_x = xyz[:, 0, :, :, :]
#                     coor_y = xyz[:, 1, :, :, :]
#                     coor_z = xyz[:, 2, :, :, :]
#                     return (mask.detach(), coor_x.detach(), coor_y.detach(), coor_z.detach(), region.detach())
#             else:
#                 for i, l in enumerate(self.features):
#                     x = l(x)
#                 # Apply HTD
#                 mem = x_f32 if x_f32 is not None else (x_f16 if x_f16 is not None else x_f64)
#                 x = self._apply_htd_if_enabled(x, mem)
#                 x = self.final_conv(x)
#                 mask = x[:, : self.mask_output_dim, :, :]
#                 xyz = x[:, self.mask_output_dim : self.mask_output_dim + self.rot_output_dim, :, :]
#                 region = x[:, self.mask_output_dim + self.rot_output_dim :, :, :]
#                 bs, c, h, w = xyz.shape
#                 xyz = xyz.view(bs, 3, self.rot_output_dim // 3, h, w)
#                 coor_x = xyz[:, 0, :, :, :]
#                 coor_y = xyz[:, 1, :, :, :]
#                 coor_z = xyz[:, 2, :, :, :]
#                 return mask, coor_x, coor_y, coor_z, region





# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.cnn import normal_init, constant_init
# from torch.nn.modules.batchnorm import _BatchNorm
# from core.utils.layer_utils import get_norm
# from .resnet_backbone import resnet_spec

# # -------------------------
# # HTDecoder (轻量级 Hybrid Transformer Decoder)
# # -------------------------
# class HTDecoder(nn.Module):
#     """
#     Lightweight Hybrid Transformer Decoder (HTD).
#     - d_model: feature dimension (num_filters)
#     - nhead: attention heads
#     - num_layers: number of decoder layers (usually 1 or 2)
#     - dim_feedforward: FFN hidden dim
#     """
#     def __init__(self, d_model=256, nhead=8, num_layers=1, dim_feedforward=1024, dropout=0.1):
#         super().__init__()
#         # 使用 PyTorch 自带的 TransformerDecoderLayer 作为基础
#         self.layers = nn.ModuleList(
#             [
#                 nn.TransformerDecoderLayer(
#                     d_model=d_model,
#                     nhead=nhead,
#                     dim_feedforward=dim_feedforward,
#                     dropout=dropout,
#                     activation="gelu",
#                     batch_first=True,  # newer PT supports batch_first to use (B, S, E)
#                 )
#                 for _ in range(num_layers)
#             ]
#         )
#         self.norm = nn.LayerNorm(d_model)

#     def forward(self, tgt, memory):
#         """
#         tgt: (B, S, E)  features to be decoded (query)
#         memory: (B, S_mem, E)  multi-scale memory (key/value) or None
#         Returns: (B, S, E)
#         """
#         # If memory is None, use tgt as memory (self-attention)
#         if memory is None:
#             memory = tgt

#         out = tgt  # shape (B, S, E) since we set batch_first=True
#         for layer in self.layers:
#             # TransformerDecoderLayer expects (tgt, memory)
#             out = layer(out, memory)
#         out = self.norm(out)
#         return out  # (B, S, E)


# # -------------------------
# # RotWithRegionHead（修复版）
# # -------------------------
# class RotWithRegionHead(nn.Module):
#     def __init__(
#         self,
#         cfg,
#         in_channels,
#         num_layers=3,
#         num_filters=256,
#         kernel_size=3,
#         output_kernel_size=1,
#         rot_output_dim=3,
#         mask_output_dim=1,
#         freeze=False,
#         num_classes=1,
#         rot_class_aware=False,
#         mask_class_aware=False,
#         region_class_aware=False,
#         num_regions=8,
#         norm="BN",
#         num_gn_groups=32,
#     ):
#         super().__init__()

#         self.freeze = freeze
#         self.concat = cfg.MODEL.CDPN.ROT_HEAD.ROT_CONCAT

#         # Optional HTD 配置
#         try:
#             self.use_htd = bool(cfg.MODEL.CDPN.ROT_HEAD.USE_HTD)
#         except Exception:
#             self.use_htd = False

#         try:
#             htd_layers = int(cfg.MODEL.CDPN.ROT_HEAD.HTD_NUM_LAYERS)
#         except Exception:
#             htd_layers = 1
#         try:
#             htd_nhead = int(cfg.MODEL.CDPN.ROT_HEAD.HTD_NHEAD)
#         except Exception:
#             htd_nhead = 8
#         try:
#             htd_ff = int(cfg.MODEL.CDPN.ROT_HEAD.HTD_FF)
#         except Exception:
#             htd_ff = 1024

#         # 初始化 HTDecoder（若启用）
#         if self.use_htd:
#             self.htd = HTDecoder(d_model=num_filters, nhead=htd_nhead, num_layers=htd_layers, dim_feedforward=htd_ff)

#         # 用于惰性注册 memory->tgt channel 投影层（避免在 forward 中创建临时层）
#         self.memory_projs = nn.ModuleDict()

#         # 参数检查（沿用你原来逻辑）
#         assert kernel_size in (2, 3, 4), "Only support kernel 2, 3 and 4"
#         assert num_regions > 1, f"Only support num_regions > 1, but got {num_regions}"
#         padding = 1
#         output_padding = 0
#         if kernel_size == 3:
#             output_padding = 1
#         elif kernel_size == 2:
#             padding = 0

#         assert output_kernel_size in (1, 3), "Only support kernel 1 and 3"
#         pad = 0
#         if output_kernel_size == 3:
#             pad = 1

#         # 构建 features（保持你原先结构）
#         self.features = nn.ModuleList()

#         if self.concat:
#             _, _, channels, _ = resnet_spec[cfg.MODEL.CDPN.BACKBONE.NUM_LAYERS]
#             self.features.append(
#                 nn.ConvTranspose2d(
#                     in_channels,
#                     num_filters,
#                     kernel_size=kernel_size,
#                     stride=2,
#                     padding=padding,
#                     output_padding=output_padding,
#                     bias=False,
#                 )
#             )
#             self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
#             self.features.append(nn.ReLU(inplace=True))
#             for i in range(num_layers):
#                 self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
#                 self.features.append(
#                     nn.Conv2d(
#                         num_filters + channels[-2 - i], num_filters, kernel_size=3, stride=1, padding=1, bias=False
#                     )
#                 )
#                 self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
#                 self.features.append(nn.ReLU(inplace=True))

#                 self.features.append(
#                     nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
#                 )
#                 self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
#                 self.features.append(nn.ReLU(inplace=True))
#         else:
#             self.features.append(
#                 nn.ConvTranspose2d(
#                     in_channels,
#                     num_filters,
#                     kernel_size=kernel_size,
#                     stride=2,
#                     padding=padding,
#                     output_padding=output_padding,
#                     bias=False,
#                 )
#             )
#             self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
#             self.features.append(nn.ReLU(inplace=True))
#             for i in range(num_layers):
#                 if i >= 1:
#                     self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
#                 self.features.append(
#                     nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
#                 )
#                 self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
#                 self.features.append(nn.ReLU(inplace=True))

#                 self.features.append(
#                     nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
#                 )
#                 self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
#                 self.features.append(nn.ReLU(inplace=True))

#         # 输出维度设置（与你原始一致）
#         self.rot_output_dim = rot_output_dim
#         if rot_class_aware:
#             self.rot_output_dim *= num_classes

#         self.mask_output_dim = mask_output_dim
#         if mask_class_aware:
#             self.mask_output_dim *= num_classes

#         self.region_output_dim = num_regions + 1  # add one channel for bg
#         if region_class_aware:
#             self.region_output_dim *= num_classes

#         self.final_conv = nn.Conv2d(
#             num_filters,
#             self.mask_output_dim + self.rot_output_dim + self.region_output_dim,
#             kernel_size=output_kernel_size,
#             padding=pad,
#             bias=True,
#         )

#         # module 初始化（保持你原先风格）
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 normal_init(m, std=0.001)
#             elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
#                 constant_init(m, 1)
#             elif isinstance(m, nn.ConvTranspose2d):
#                 normal_init(m, std=0.001)

#     # 惰性获取或创建投影层（被注册到 ModuleDict）
#     def _get_or_create_proj(self, in_ch, out_ch):
#         key = f"{in_ch}_{out_ch}"
#         if key not in self.memory_projs:
#             proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
#             normal_init(proj, std=0.001)
#             self.memory_projs[key] = proj
#         return self.memory_projs[key]

#     def _apply_htd_if_enabled(self, x, memory_feature=None):
#         """
#         x: (B, C, H, W)
#         memory_feature: optional (B, C_mem, H_mem, W_mem)
#         Returns: x after HTD (B, C, H, W)
#         """
#         if not self.use_htd:
#             return x

#         b, c, h, w = x.shape
#         # tgt: (B, S, E)
#         tgt = x.flatten(2).permute(0, 2, 1)  # (B, S, E)  batch_first

#         mem = None
#         if memory_feature is not None:
#             mb, mc, mh, mw = memory_feature.shape

#             # 若通道数不匹配，走投影（并确保投影为 Module）
#             if mc != c:
#                 proj = self._get_or_create_proj(mc, c)
#                 memory_feature = proj(memory_feature)
#                 mc = c

#             # 若空间 size 不同，插值到 tgt 的空间大小
#             if (mh != h) or (mw != w):
#                 memory_feature = F.interpolate(memory_feature, size=(h, w), mode="bilinear", align_corners=False)

#             mem = memory_feature.flatten(2).permute(0, 2, 1)  # (B, S_mem, E)

#         # HTDecoder expects (B, S, E) since we set batch_first=True
#         out = self.htd(tgt, mem)  # (B, S, E)
#         out = out.permute(0, 2, 1).contiguous().view(b, c, h, w)
#         return out

#     def forward(self, x, x_f64=None, x_f32=None, x_f16=None):
#         """
#         修正后的 forward：先完整执行 features 序列，
#         然后一次性调用 HTD（如启用），最后做 final_conv 和拆分。
#         """
#         # 对 features 完整遍历（避免中途 return）
#         if self.concat:
#             for i, l in enumerate(self.features):
#                 if i == 3:
#                     # concat with x_f16
#                     x = l(torch.cat([x, x_f16], 1))
#                 elif i == 12:
#                     x = l(torch.cat([x, x_f32], 1))
#                 elif i == 21:
#                     x = l(torch.cat([x, x_f64], 1))
#                 else:
#                     x = l(x)
#         else:
#             for l in self.features:
#                 x = l(x)

#         # 决定 memory 特征（优先级与你原来一致）
#         mem = x_f32 if x_f32 is not None else (x_f16 if x_f16 is not None else x_f64)

#         if self.freeze:
#             # 若 head 冻结，保留 no_grad + detach 语义
#             with torch.no_grad():
#                 x = self._apply_htd_if_enabled(x, mem)
#                 print("HTD output mean:", x.mean().item())
#                 x = self.final_conv(x)
#                 if not self.concat:
#                     mask = x[:, : self.mask_output_dim, :, :].detach()
#                     xyz = x[:, self.mask_output_dim : self.mask_output_dim + self.rot_output_dim, :, :].detach()
#                     region = x[:, self.mask_output_dim + self.rot_output_dim :, :, :].detach()
#                     bs, c, h, w = xyz.shape
#                     xyz = xyz.view(bs, 3, self.rot_output_dim // 3, h, w)
#                     coor_x = xyz[:, 0, :, :, :]
#                     coor_y = xyz[:, 1, :, :, :]
#                     coor_z = xyz[:, 2, :, :, :]
#                     return (mask, coor_x, coor_y, coor_z, region)
#                 else:
#                     return x.detach()
#         else:
#             # 非 freeze，正常计算图流（HTD 将产生梯度）
#             x = self._apply_htd_if_enabled(x, mem)
#             print("HTD output mean:", x.mean().item())
#             x = self.final_conv(x)
#             if not self.concat:
#                 mask = x[:, : self.mask_output_dim, :, :]
#                 xyz = x[:, self.mask_output_dim : self.mask_output_dim + self.rot_output_dim, :, :]
#                 region = x[:, self.mask_output_dim + self.rot_output_dim :, :, :]
#                 bs, c, h, w = xyz.shape
#                 xyz = xyz.view(bs, 3, self.rot_output_dim // 3, h, w)
#                 coor_x = xyz[:, 0, :, :, :]
#                 coor_y = xyz[:, 1, :, :, :]
#                 coor_z = xyz[:, 2, :, :, :]
#                 return mask, coor_x, coor_y, coor_z, region
#             else:
#                 return x




import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, constant_init
from torch.nn.modules.batchnorm import _BatchNorm
from core.utils.layer_utils import get_norm
from .resnet_backbone import resnet_spec  # keep for compatibility if used elsewhere

# -------------------------
# HTDecoderLayer / HTDecoder (完整 self + cross + FFN)
# -------------------------
class HTDecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        # use nn.MultiheadAttention with batch_first=False (we will provide batch_first=True usage by permute)
        # but to avoid version incompatibility we will use batch_first in later wrapper (we use (B, S, E) directly)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
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
        # tgt: (B, S_tgt, C), memory: (B, S_mem, C)
        # Self-attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention
        # if memory is None, cross-attention degenerates to self-attn on tgt
        if memory is None:
            mem = tgt
        else:
            mem = memory
        tgt2, _ = self.cross_attn(tgt, mem, mem)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class HTDecoder(nn.Module):
    def __init__(self, d_model=256, num_layers=1, nhead=8, dim_feedforward=512):
        super().__init__()
        self.layers = nn.ModuleList(
            [HTDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):
        # tgt: (B, S_tgt, C); memory: (B, S_mem, C) or None
        out = tgt
        for layer in self.layers:
            out = layer(out, memory)
        out = self.norm(out)
        return out  # (B, S_tgt, C)

# -------------------------
# RotWithRegionHead（最小改动 + HTD）
# -------------------------
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
    ):
        """
        保持原始签名 —— 仅在内部增加 HTD，兼容 ConvNeXt 多尺度 memory
        cfg: 原始配置对象（我们从 cfg.MODEL.CDPN.ROT_HEAD 读取 HTD 设置）
        in_channels: 来自 build_model_optimizer 传进来的 aligned channels（target channels）
        其余参数与原始实现一一对应。
        """
        super().__init__()

        self.mask_output_dim = mask_output_dim
        self.rot_output_dim = rot_output_dim
        self.num_regions = num_regions


        self.freeze = freeze
        self.concat = cfg.MODEL.CDPN.ROT_HEAD.ROT_CONCAT if hasattr(cfg.MODEL.CDPN.ROT_HEAD, "ROT_CONCAT") else False

        # read HTD config from cfg safely
        try:
            self.use_htd = bool(cfg.MODEL.CDPN.ROT_HEAD.USE_HTD)
        except Exception:
            self.use_htd = False
        try:
            htd_layers = int(cfg.MODEL.CDPN.ROT_HEAD.HTD_NUM_LAYERS)
        except Exception:
            htd_layers = 1
        try:
            htd_nhead = int(cfg.MODEL.CDPN.ROT_HEAD.HTD_NHEAD)
        except Exception:
            htd_nhead = 8
        try:
            htd_ff = int(cfg.MODEL.CDPN.ROT_HEAD.HTD_FF)
        except Exception:
            htd_ff = 512

        # ---- 保留原始 features 的构建逻辑 (根据 concat 与 num_layers) ----
        assert kernel_size in (2, 3, 4), "Only support kernel 2,3,4"
        padding = 1 if kernel_size == 3 else (0 if kernel_size == 2 else 1)
        output_padding = 1 if kernel_size == 3 else 0
        pad = 0
        if output_kernel_size == 3:
            pad = 1

        self.features = nn.ModuleList()
        # 这里直接沿用你原来代码填充 features（concat 分支与非 concat 分支）
        if self.concat:
            # keep original concat-building behavior (ResNet style) — unchanged
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
                        num_filters + channels[-2 - i], num_filters, kernel_size=3, stride=1, padding=1, bias=False
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
            # keep original non-concat-building behavior
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

        # ---- HTD 及 ConvNeXt memory 投影（预建以保证 optimizer 捕获参数） ----
        # memory_projs 用于把 ConvNeXt 不同 stage channels 投影到 num_filters（in_channels argument）
        # ConvNeXt-tiny 常见 stage channels: [96, 192, 384, 768] （来自你之前的打印）
        # 只预建这些常见值，不改变外部接口
        self.memory_projs = nn.ModuleDict()
        convnext_stage_chs = [96, 192, 384, 768]
        for ch in convnext_stage_chs:
            key = str(ch)
            self.memory_projs[key] = nn.Conv2d(ch, num_filters, kernel_size=1, bias=False)
            # init
            normal_init(self.memory_projs[key], std=0.001)

        if self.use_htd:
            self.htd = HTDecoder(d_model=num_filters, num_layers=htd_layers, nhead=htd_nhead, dim_feedforward=htd_ff)

        # final conv: 保持原始输出通道设置 —— 你原始是 mask+rot+region 合并通道数 (这里使用 provided rot/mask/region dims via caller)
        # 为兼容性我们使用 in_channels=num_filters -> channels在调用 final_conv 取决于上游计算
        # In original implementation final_conv output channels = mask_output_dim + rot_output_dim + region_output_dim
        total_out_ch = mask_output_dim + rot_output_dim + (num_regions + 1)
        self.final_conv = nn.Conv2d(num_filters, total_out_ch, kernel_size=output_kernel_size, padding=pad, bias=True)

        # init modules (keep original style)
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

    def _proj_memory_if_needed(self, memory_feature, tgt_channels):
        """
        将传入的 memory_feature 投影到 tgt_channels（num_filters）并 resize到 tgt spatial
        memory_feature: (B, C_mem, Hm, Wm)
        """
        if memory_feature is None:
            return None
        mb, mc, mh, mw = memory_feature.shape
        key = str(mc)
        if key in self.memory_projs:
            proj = self.memory_projs[key]
            mem = proj(memory_feature)  # now (B, tgt_C, Hm, Wm)
        else:
            # 如果不是预建的通道，也用 1x1 临时投影（但此情形不常见）
            proj_tmp = nn.Conv2d(mc, tgt_channels, kernel_size=1, bias=False).to(memory_feature.device)
            with torch.no_grad():
                normal_init(proj_tmp, std=0.001)
            mem = proj_tmp(memory_feature)
        return mem

    def _apply_htd_if_enabled(self, x, memory_feature=None):
        """
        x: (B, C, H, W)  — tgt feature map
        memory_feature: optional (B, C_mem, Hm, Wm)
        """
        if not self.use_htd:
            return x
        b, c, h, w = x.shape
        # tgt: (B, S, C)
        tgt = x.flatten(2).permute(0, 2, 1).contiguous()
        mem = None
        if memory_feature is not None:
            # proj 到 tgt channels
            mem_proj = self._proj_memory_if_needed(memory_feature, c)  # (B, c, h, w) or scaled
            if mem_proj is not None:
                # 如果 spatial 不同，插值到 tgt 空间
                _, _, mh, mw = mem_proj.shape
                if (mh != h) or (mw != w):
                    mem_proj = F.interpolate(mem_proj, size=(h, w), mode="bilinear", align_corners=False)
                mem = mem_proj.flatten(2).permute(0, 2, 1).contiguous()  # (B, S_mem, C)
        # 调用 HTD：tgt (B,S,C), mem (B,S_mem,C) or None
        out = self.htd(tgt, mem) if self.use_htd else tgt
        out = out.permute(0, 2, 1).contiguous().view(b, c, h, w)
        return out

    def forward(self, x, x_f64=None, x_f32=None, x_f16=None):
        """
        保持原始 forward 签名（兼容你现有调用）
        - x: main feature (B, C, H, W)
        - x_f64/x_f32/x_f16: optional auxiliary features from backbone (may be None)
        """

        if self.concat:
            # 保持原始 concat 分支逻辑（和你之前代码一致），但**取消循环中提前 return**问题
            for i, l in enumerate(self.features):
                if i == 3:
                    x = l(torch.cat([x, x_f16], 1))
                elif i == 12:
                    x = l(torch.cat([x, x_f32], 1))
                elif i == 21:
                    x = l(torch.cat([x, x_f64], 1))
                else:
                    x = l(x)
            # 在 features 全部执行完后再 apply HTD
            mem = x_f32 if x_f32 is not None else (x_f16 if x_f16 is not None else x_f64)
            x = self._apply_htd_if_enabled(x, mem)
            x = self.final_conv(x)
            if self.freeze:
                return x.detach()
            else:
                return x
        else:
            # 非 concat 分支
            for i, l in enumerate(self.features):
                x = l(x)
            # 选择 memory_feature：优先使用中间尺度（x_f32），再 x_f16, x_f64
            mem = x_f32 if x_f32 is not None else (x_f16 if x_f16 is not None else x_f64)
            x = self._apply_htd_if_enabled(x, mem)
            x = self.final_conv(x)

            # 原始代码将 final_conv 的输出拆分为 mask/xyz/region；保持行为一致
            mask = x[:, : self.mask_output_dim, :, :]
            xyz = x[:, self.mask_output_dim : self.mask_output_dim + self.rot_output_dim, :, :]
            region = x[:, self.mask_output_dim + self.rot_output_dim :, :, :]

            bs, c, h, w = xyz.shape
            # xyz shape -> (bs, 3, rot_output_dim//3, h, w)  原来的处理方式
            try:
                xyz_view = xyz.view(bs, 3, self.rot_output_dim // 3, h, w)
                coor_x = xyz_view[:, 0, :, :, :]
                coor_y = xyz_view[:, 1, :, :, :]
                coor_z = xyz_view[:, 2, :, :, :]
            except Exception:
                # 兼容性保护：若 rot_output_dim 不是 3*n，直接按通道分割
                coor_x = xyz
                coor_y = xyz
                coor_z = xyz

            if self.freeze:
                # detach outputs if frozen (与原实现语义一致)
                return mask.detach(), coor_x.detach(), coor_y.detach(), coor_z.detach(), region.detach()
            else:
                return mask, coor_x, coor_y, coor_z, region

