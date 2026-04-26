# convnext_with_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Tuple

# -----------------------------
# Positional encoding helper
# -----------------------------
def get_2d_sincos_pos_emb(H: int, W: int, C: int, device: torch.device) -> torch.Tensor:
    """
    Produce a (H*W, C) 2D sin-cos positional embedding.
    C must be even. Non-learned, deterministic.
    """
    assert C % 2 == 0, "Channel dim for sincos pos emb must be even"
    y = torch.arange(H, dtype=torch.float32, device=device)
    x = torch.arange(W, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # H, W
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # S,2 (x,y)

    dim_half = C // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_half, 2, device=device).float() / dim_half))
    # build for x and y separately and concat
    pe_x = coords[:, 0:1] * inv_freq.unsqueeze(0)  # S, dim_half/?
    pe_y = coords[:, 1:2] * inv_freq.unsqueeze(0)
    # for each we interleave sin/cos -> final size dim_half
    sincos_x = torch.cat([torch.sin(pe_x), torch.cos(pe_x)], dim=1)  # S, dim_half
    sincos_y = torch.cat([torch.sin(pe_y), torch.cos(pe_y)], dim=1)  # S, dim_half
    pe = torch.cat([sincos_x, sincos_y], dim=1)  # S, C
    return pe  # (S, C)


# -----------------------------
# TransformerBlock2D
# -----------------------------
class TransformerBlock2D(nn.Module):
    """
    2D Transformer block that accepts (B, C, H, W).
    - pre-norm style
    - optional positional encoding (sincos)
    - optional cross-attention with memory (B, C_mem, Hm, Wm) projected to C via conv1x1
    - returns transformed (B, C, H, W). Optionally returns attention weights if return_attn=True.
    """
    def __init__(
        self,
        in_channels: int,
        feedforward_dim: int = 2048,
        num_heads: int = 4,
        drop: float = 0.0,
        use_pos_emb: bool = True,
        attn_dropout: float = 0.0,
        use_cross_attn: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.use_pos_emb = use_pos_emb
        self.use_cross_attn = use_cross_attn

        # LayerNorm will be applied on last dim (C)
        self.norm1 = nn.LayerNorm(in_channels)
        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True, dropout=attn_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()

        if self.use_cross_attn:
            # cross-attn: query from tgt, key/value from memory (same embed dim)
            self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True, dropout=attn_dropout)
            # if memory channels differ, user should project externally or we provide a projection helper (1x1 conv registered)
            # We'll create a generic conv_proj that can be replaced if needed
            self.memory_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
            nn.init.normal_(self.memory_proj.weight, std=0.001)

        self.norm2 = nn.LayerNorm(in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, feedforward_dim),
            nn.GELU(),
            nn.Dropout(drop) if drop > 0 else nn.Identity(),
            nn.Linear(feedforward_dim, in_channels),
            nn.Dropout(drop) if drop > 0 else nn.Identity(),
        )

        # cache for pos emb
        self._pos_cache = {}

    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None, return_attn: bool = False):
        """
        x: (B, C, H, W)
        memory (optional): (B, C_mem, Hm, Wm) OR (B, C, H, W) if already same channels.
        return_attn: if True, return (out, attn_weights) where attn_weights is from self-attn.
        """
        B, C, H, W = x.shape
        S = H * W
        device = x.device

        x_flat = x.flatten(2).transpose(1, 2).contiguous()  # (B, S, C)

        # positional encoding
        if self.use_pos_emb:
            key = (H, W, C)
            if key not in self._pos_cache or self._pos_cache[key].device != device:
                self._pos_cache[key] = get_2d_sincos_pos_emb(H, W, C, device)
            pos = self._pos_cache[key].unsqueeze(0).expand(B, -1, -1)  # (B, S, C)
            x_flat = x_flat + pos

        # pre-norm once
        normed = self.norm1(x_flat)  # (B, S, C)

        # self-attention (Q=K=V=normed)
        self_attn_out, self_attn_w = self.self_attn(normed, normed, normed, need_weights=True)
        self_attn_out = self.attn_dropout(self_attn_out)
        x_res = x_flat + self_attn_out  # residual after self-attn

        # optional cross-attention
        cross_attn_w = None
        if self.use_cross_attn and memory is not None:
            # memory may have different spatial resolution; project to same channel dim and interpolate
            # memory expected shape: (B, C_mem, Hm, Wm)
            mem = memory
            if mem.dim() == 4:
                # if channels differ, project
                if mem.shape[1] != C:
                    mem = self.memory_proj(mem)
                # interpolate to tgt spatial
                if mem.shape[2] != H or mem.shape[3] != W:
                    mem = F.interpolate(mem, size=(H, W), mode="bilinear", align_corners=False)
                mem_flat = mem.flatten(2).transpose(1, 2).contiguous()  # (B, S, C)
            else:
                raise ValueError("memory must be 4D tensor (B,C,H,W)")
            # use normed as query, mem_flat as key/value
            query = self.norm1(x_res)  # using same norm (or separate norm could be used)
            cross_out, cross_attn_w = self.cross_attn(query, mem_flat, mem_flat, need_weights=True)
            cross_out = self.attn_dropout(cross_out)
            x_res = x_res + cross_out  # residual for cross-attn

        # feed-forward
        x_ffn = self.mlp(self.norm2(x_res))
        out = x_res + x_ffn  # final residual

        out = out.transpose(1, 2).reshape(B, C, H, W).contiguous()

        if return_attn:
            return out, {"self_attn": self_attn_w, "cross_attn": cross_attn_w}
        return out


# -----------------------------
# ECALayer (Efficent Channel Attention)
# -----------------------------
class ECALayer(nn.Module):
    def __init__(self, channel: int, k_size: int = 7):
        """
        Efficient Channel Attention:
        - global avg pool -> 1D conv across channels -> sigmoid -> scale
        channel: input channels
        k_size: kernel size in conv1d (should be odd)
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # conv1d expects (B, C, L) but we will pass (B, 1, C) by permuting
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W)
        b, c, _, _ = x.shape
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.view(b, c)  # (B, C)
        # to shape (B, 1, C) for conv1d across channel dimension
        y = y.unsqueeze(1)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = self.sigmoid(y)  # (B, 1, C)
        y = y.transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y.expand_as(x)


# -----------------------------
# ConvNeXtBackboneNet with TransformerBlock2D and ECA
# -----------------------------
class ConvNeXtBackboneNet(nn.Module):
    def __init__(
        self,
        variant: str = "convnext_tiny",
        in_channel: int = 3,
        pretrained: bool = True,
        freeze: bool = False,
        rot_concat: bool = False,
        use_transformer: bool = False,
        transformer_heads: int = 4,
        transformer_feedforward_dim: int = 2048,
        transformer_dropout: float = 0.0,
        transformer_attn_dropout: float = 0.0,
        transformer_use_pos: bool = True,
        transformer_use_cross_attn: bool = False,
        use_attention: bool = False,
    ):
        """
        variant: timm convnext variant name
        rot_concat: if True, forward returns tuple (x_high, x_f64, x_f32, x_f16)
        use_transformer: apply TransformerBlock2D on top feature (x_high)
        transformer_use_cross_attn: if True, TransformerBlock2D will expect memory to be provided (we pass x_f32 as memory)
        use_attention: whether to use ECA on x_high
        """
        super().__init__()
        self.freeze = freeze
        self.rot_concat = rot_concat

        # build backbone
        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            in_chans=in_channel,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        # features_only returns list of feature maps: low->high
        # Here we expect feats = [x_f64, x_f32, x_f16, x_high] but confirm with timm model
        # Use last feature channels as out_channels
        self.out_channels = self.backbone.feature_info[-1]["num_chs"]

        # transformer block (applied to x_high)
        self.use_transformer = use_transformer
        self.transformer_block: Optional[TransformerBlock2D] = None
        if self.use_transformer:
            self.transformer_block = TransformerBlock2D(
                in_channels=self.out_channels,
                feedforward_dim=transformer_feedforward_dim,
                num_heads=transformer_heads,
                drop=transformer_dropout,
                use_pos_emb=transformer_use_pos,
                attn_dropout=transformer_attn_dropout,
                use_cross_attn=transformer_use_cross_attn,
            )

        # attention (ECA)
        self.use_attention = use_attention
        self.eca = ECALayer(self.out_channels) if self.use_attention else None

    def forward(self, x: torch.Tensor):
        """
        Return:
        - if rot_concat: (x_high, x_f64, x_f32, x_f16) (matching your previous API order)
        - else: x_high
        Notes:
        - if transformer.use_cross_attn True, we pass x_f32 as memory to transformer_block
        """
        feats = self.backbone(x)
        # timm's features_only returns a list from early to late; typical convnext: [stage0, stage1, stage2, stage3]
        # Map them to your naming: keep old variable names for compatibility
        if len(feats) >= 4:
            x_f64, x_f32, x_f16, x_high = feats[:4]
        else:
            # fallback: last is x_high, rest None
            *rest, x_high = feats
            # fill others with None
            pad = [None] * (3 - len(rest))
            rest = pad + rest
            x_f64, x_f32, x_f16 = rest

        # apply transformer on top feature
        if self.use_transformer and (self.transformer_block is not None):
            if getattr(self.transformer_block, "use_cross_attn", False):
                # supply memory as x_f32 if available, else None
                mem = x_f32 if x_f32 is not None else (x_f16 if x_f16 is not None else x_f64)
                if mem is None:
                    # If no memory available, fall back to self-attn only
                    x_high = self.transformer_block(x_high, memory=None)
                else:
                    x_high = self.transformer_block(x_high, memory=mem)
            else:
                # simple self-attention block
                x_high = self.transformer_block(x_high, memory=None)

        # apply ECA channel attention if needed
        if self.use_attention and (self.eca is not None):
            x_high = self.eca(x_high)

        # freeze logic: return detached tensors to prevent gradients if freeze True
        if self.freeze:
            with torch.no_grad():
                if self.rot_concat:
                    return x_high.detach(), x_f64.detach() if x_f64 is not None else None, x_f32.detach() if x_f32 is not None else None, x_f16.detach() if x_f16 is not None else None
                return x_high.detach()
        else:
            if self.rot_concat:
                return x_high, x_f64, x_f32, x_f16
            return x_high


# -----------------------------
# Example usage snippet (not executed here)
# -----------------------------
if __name__ == "__main__":
    # quick smoke test
    model = ConvNeXtBackboneNet(
        variant="convnext_tiny",
        in_channel=3,
        pretrained=False,
        freeze=False,
        rot_concat=True,
        use_transformer=True,
        transformer_heads=4,
        transformer_feedforward_dim=1024,
        transformer_use_cross_attn=False,
        use_attention=True,
    )
    dummy = torch.randn(2, 3, 224, 224)
    outputs = model(dummy)
    print("Output types / shapes:")
    if isinstance(outputs, tuple):
        print([o.shape if o is not None else None for o in outputs])
    else:
        print(outputs.shape)




# import torch
# import torch.nn as nn
# import timm

# class TransformerBlock2D(nn.Module):
#     def __init__(self, in_channels, feedforward_dim=2048, num_heads=4, drop=0.0):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(in_channels)
#         self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
#         self.norm2 = nn.LayerNorm(in_channels)
#         self.mlp = nn.Sequential(
#             nn.Linear(in_channels, feedforward_dim),
#             nn.GELU(),
#             nn.Linear(feedforward_dim, in_channels),
#         )

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x_flat = x.flatten(2).transpose(1, 2)  # B,HW,C
#         x_res = x_flat
#         attn_out, _ = self.attn(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat))
#         x = x_res + attn_out
#         x = x + self.mlp(self.norm2(x))
#         x = x.transpose(1, 2).reshape(B, C, H, W)
#         return x


# class ECALayer(nn.Module):
#     def __init__(self, channel, k_size=7):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(
#             1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv(y.squeeze(-1).transpose(-1, -2))
#         y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
#         return x * y.expand_as(x)


# class ConvNeXtBackboneNet(nn.Module):
#     def __init__(
#         self,
#         variant="convnext_tiny",
#         in_channel=3,
#         pretrained=True,
#         freeze=False,
#         rot_concat=False,
#         use_transformer=False,
#         transformer_heads=4,
#         transformer_feedforward_dim=2048,
#         use_attention=False,
#     ):
#         super().__init__()
#         self.freeze = freeze
#         self.rot_concat = rot_concat
#         self.backbone = timm.create_model(
#             variant,
#             pretrained=pretrained,
#             in_chans=in_channel,
#             features_only=True,
#             out_indices=(0, 1, 2, 3),
#         )
#         self.out_channels = self.backbone.feature_info[-1]["num_chs"]

#         # 仅定义一个Transformer块，移除trans_block变量
#         self.use_transformer = use_transformer
#         self.transformer_block = None
#         if self.use_transformer:
#             self.transformer_block = TransformerBlock2D(
#                 in_channels=self.out_channels,
#                 num_heads=transformer_heads,
#                 feedforward_dim=transformer_feedforward_dim,
#             )

#         self.use_attention = use_attention
#         self.eca = ECALayer(self.out_channels) if self.use_attention else None

#     def forward(self, x: torch.Tensor):
#         feats = self.backbone(x)
#         x_f64, x_f32, x_f16, x_high = feats

#         # 直接使用transformer_block，不再使用trans_block
#         if self.use_transformer and self.transformer_block is not None:
#             x_high = self.transformer_block(x_high)

#         if self.use_attention and self.eca is not None:
#             x_high = self.eca(x_high)

#         if self.freeze:
#             with torch.no_grad():
#                 if self.rot_concat:
#                     return x_high.detach(), x_f64.detach(), x_f32.detach(), x_f16.detach()
#                 return x_high.detach()
#         else:
#             if self.rot_concat:
#                 return x_high, x_f64, x_f32, x_f16
#             return x_high





# # ===== core/gdrn_modeling/models/convnext_backbone.py =====
# import torch
# import torch.nn as nn
# import timm

# # -------------------------
# # 🔹 Transformer Block for 2D feature maps
# # -------------------------
# class TransformerBlock2D(nn.Module):
#     def __init__(self, in_channels, feedforward_dim=2048, num_heads=4, drop=0.0):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(in_channels)
#         self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
#         self.norm2 = nn.LayerNorm(in_channels)
#         self.mlp = nn.Sequential(
#             nn.Linear(in_channels, feedforward_dim),
#             nn.GELU(),
#             nn.Linear(feedforward_dim, in_channels),
#         )

#     def forward(self, x):
#         # 输入: B,C,H,W
#         B, C, H, W = x.shape
#         x_flat = x.flatten(2).transpose(1, 2)  # B,HW,C
#         x_res = x_flat
#         attn_out, _ = self.attn(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat))
#         x = x_res + attn_out
#         x = x + self.mlp(self.norm2(x))
#         x = x.transpose(1, 2).reshape(B, C, H, W)
#         return x


# # -------------------------
# # 🔹 ECA 注意力模块
# # -------------------------
# class ECALayer(nn.Module):
#     def __init__(self, channel, k_size=3):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(
#             1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         y = self.avg_pool(x)  # B,C,1,1
#         y = self.conv(y.squeeze(-1).transpose(-1, -2))
#         y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
#         return x * y.expand_as(x)


# # -------------------------
# # 🔹 ConvNeXt Backbone with optional Transformer & Attention
# # -------------------------
# class ConvNeXtBackboneNet(nn.Module):
#     def __init__(
#         self,
#         variant="convnext_tiny",
#         in_channel=3,
#         pretrained=True,
#         freeze=False,
#         rot_concat=False,
#         use_transformer=False,
#         transformer_heads=4,
#         transformer_feedforward_dim=2048,
#         use_attention=False,
#     ):
#         super().__init__()
#         self.freeze = freeze
#         self.rot_concat = rot_concat  # 添加这一行
#         # 1. 先创建主干网络
#         self.backbone = timm.create_model(
#             variant,
#             pretrained=pretrained,
#             in_chans=in_channel,
#             features_only=True,
#             out_indices=(0, 1, 2, 3),
#         )
#         # 2. 先获取输出通道数
#         self.out_channels = self.backbone.feature_info[-1]["num_chs"]

#         # 3. 再初始化 transformer block
#         self.use_transformer = use_transformer
#         if self.use_transformer:
#             self.transformer_block = TransformerBlock2D(
#                 in_channels=self.out_channels,
#                 num_heads=transformer_heads,
#                 feedforward_dim=transformer_feedforward_dim,
#             )
#         self.trans_block = getattr(self, "transformer_block", None)
#         self.use_attention = use_attention

#         # -------------------------
#         # 🔹 可选 Transformer 和注意力模块
#         # -------------------------
#         if self.use_transformer:
#             self.transformer_block = TransformerBlock2D(
#                 in_channels=self.out_channels,
#                 num_heads=transformer_heads,
#                 feedforward_dim=transformer_feedforward_dim,
#             )
#         if self.use_attention:
#             self.eca = ECALayer(self.out_channels)

#     def forward(self, x: torch.Tensor):
#         feats = self.backbone(x)  # list of 4 tensors: strides [4,8,16,32]
#         x_f64, x_f32, x_f16, x_high = feats

#         if self.use_transformer and self.trans_block is not None:
#             x_high = self.trans_block(x_high)
#             #print("Transformer block executed")
#         if self.use_attention:
#             x_high = self.eca(x_high)
#             #print("Attention block executed")

#         if self.freeze:
#             with torch.no_grad():
#                 if self.rot_concat:
#                     return x_high.detach(), x_f64.detach(), x_f32.detach(), x_f16.detach()
#                 return x_high.detach()
#         else:
#             if self.rot_concat:
#                 return x_high, x_f64, x_f32, x_f16
#             return x_high




# # ===== NEW FILE =====
# # core/gdrn_modeling/models/convnext_backbone.py
# import torch
# import torch.nn as nn

# try:
#     from timm import create_model
# except Exception as e:  # pragma: no cover
#     raise ImportError("convnext_backbone requires 'timm'. Install: pip install timm")


# class ConvNeXtBackboneNet(nn.Module):
#     """
#     ConvNeXt backbone wrapper using timm with features_only=True.
#     Returns final stage feature (stride 32) and optionally low-level features
#     in the same order expected by current GDRN (x_f64, x_f32, x_f16).
#     """

#     def __init__(
#         self,
#         variant: str = "convnext_tiny",  # convnext_tiny/small/base/large
#         in_channel: int = 3,
#         pretrained: bool = True,
#         freeze: bool = False,
#         rot_concat: bool = False,
#     ):
#         super().__init__()
#         self.freeze = freeze
#         self.rot_concat = rot_concat

#         self.backbone = create_model(
#             variant,
#             pretrained=pretrained,
#             in_chans=in_channel,
#             features_only=True,
#             out_indices=(0, 1, 2, 3),  # strides: [4, 8, 16, 32]
#         )
#         # timm FeatureInfo entries contain num_chs
#         self.out_channels = self.backbone.feature_info[-1]["num_chs"]

#     def forward(self, x: torch.Tensor):
#         feats = self.backbone(x)  # list of 4 tensors at strides [4,8,16,32]
#         x_f64, x_f32, x_f16, x_high = feats  # keep naming consistent with existing heads
#         if self.freeze:
#             with torch.no_grad():
#                 if self.rot_concat:
#                     return x_high.detach(), x_f64.detach(), x_f32.detach(), x_f16.detach()
#                 return x_high.detach()
#         else:
#             if self.rot_concat:
#                 return x_high, x_f64, x_f32, x_f16
#             return x_high