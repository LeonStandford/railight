from __future__ import annotations
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import LayerNorm2d
from ultralytics.utils.instance import to_2tuple


class Conv2d_BN(torch.nn.Sequential):

    def __init__(
        self,
        a: int,
        b: int,
        ks: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bn_weight_init: float = 1,
    ):
        super().__init__()
        self.add_module(
            "c", torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False)
        )
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)


class PatchEmbed(nn.Module):

    def __init__(self, in_chans: int, embed_dim: int, resolution: int, activation):
        super().__init__()
        img_size: tuple[int, int] = to_2tuple(resolution)
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class MBConv(nn.Module):

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        expand_ratio: float,
        activation,
        drop_path: float,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans
        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()
        self.conv2 = Conv2d_BN(
            self.hidden_chans,
            self.hidden_chans,
            ks=3,
            stride=1,
            pad=1,
            groups=self.hidden_chans,
        )
        self.act2 = activation()
        self.conv3 = Conv2d_BN(self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()
        self.drop_path = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        x += shortcut
        return self.act3(x)


class PatchMerging(nn.Module):

    def __init__(
        self, input_resolution: tuple[int, int], dim: int, out_dim: int, activation
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        stride_c = 1 if out_dim in {320, 448, 576} else 2
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        return x.flatten(2).transpose(1, 2)


class ConvLayer(nn.Module):

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        activation,
        drop_path: float | list[float] = 0.0,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False,
        out_dim: int | None = None,
        conv_expand_ratio: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                MBConv(
                    dim,
                    dim,
                    conv_expand_ratio,
                    activation,
                    drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(depth)
            ]
        )
        self.downsample = (
            None
            if downsample is None
            else downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = torch.utils.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        return x if self.downsample is None else self.downsample(x)


class MLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        activation=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = activation()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int = 8,
        attn_ratio: float = 4,
        resolution: tuple[int, int] = (14, 14),
    ):
        super().__init__()
        assert (
            isinstance(resolution, tuple) and len(resolution) == 2
        ), "'resolution' argument not tuple of length 2"
        self.num_heads = num_heads
        self.scale = key_dim ** (-0.5)
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)
        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets))
        )
        self.register_buffer(
            "attention_bias_idxs", torch.LongTensor(idxs).view(N, N), persistent=False
        )

    @torch.no_grad()
    def train(self, mode: bool = True):
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.d], dim=3
        )
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        self.ab = self.ab.to(self.attention_biases.device)
        attn = q @ k.transpose(-2, -1) * self.scale + (
            self.attention_biases[:, self.attention_bias_idxs]
            if self.training
            else self.ab
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        return self.proj(x)


class TinyViTBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        local_conv_size: int = 3,
        activation=nn.GELU,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, "window_size must be greater than 0"
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.drop_path = nn.Identity()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads
        window_resolution = (window_size, window_size)
        self.attn = Attention(
            dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            activation=mlp_activation,
            drop=drop,
        )
        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.input_resolution
        b, hw, c = x.shape
        assert hw == h * w, "input feature has wrong size"
        res_x = x
        if h == self.window_size and w == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(b, h, w, c)
            pad_b = (self.window_size - h % self.window_size) % self.window_size
            pad_r = (self.window_size - w % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0
            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            pH, pW = (h + pad_b, w + pad_r)
            nH = pH // self.window_size
            nW = pW // self.window_size
            x = (
                x.view(b, nH, self.window_size, nW, self.window_size, c)
                .transpose(2, 3)
                .reshape(b * nH * nW, self.window_size * self.window_size, c)
            )
            x = self.attn(x)
            x = (
                x.view(b, nH, nW, self.window_size, self.window_size, c)
                .transpose(2, 3)
                .reshape(b, pH, pW, c)
            )
            if padding:
                x = x[:, :h, :w].contiguous()
            x = x.view(b, hw, c)
        x = res_x + self.drop_path(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.local_conv(x)
        x = x.view(b, c, hw).transpose(1, 2)
        return x + self.drop_path(self.mlp(x))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"


class BasicLayer(nn.Module):

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False,
        local_conv_size: int = 3,
        activation=nn.GELU,
        out_dim: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                TinyViTBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    local_conv_size=local_conv_size,
                    activation=activation,
                )
                for i in range(depth)
            ]
        )
        self.downsample = (
            None
            if downsample is None
            else downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = torch.utils.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        return x if self.downsample is None else self.downsample(x)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class TinyViT(nn.Module):

    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dims: tuple[int, int, int, int] = (96, 192, 384, 768),
        depths: tuple[int, int, int, int] = (2, 2, 6, 2),
        num_heads: tuple[int, int, int, int] = (3, 6, 12, 24),
        window_sizes: tuple[int, int, int, int] = (7, 7, 14, 7),
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False,
        mbconv_expand_ratio: float = 4.0,
        local_conv_size: int = 3,
        layer_lr_decay: float = 1.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        activation = nn.GELU
        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            resolution=img_size,
            activation=activation,
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs = dict(
                dim=embed_dims[i_layer],
                input_resolution=(
                    patches_resolution[0]
                    // 2 ** (i_layer - 1 if i_layer == 3 else i_layer),
                    patches_resolution[1]
                    // 2 ** (i_layer - 1 if i_layer == 3 else i_layer),
                ),
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
                out_dim=embed_dims[min(i_layer + 1, len(embed_dims) - 1)],
                activation=activation,
            )
            if i_layer == 0:
                layer = ConvLayer(conv_expand_ratio=mbconv_expand_ratio, **kwargs)
            else:
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs,
                )
            self.layers.append(layer)
        self.norm_head = nn.LayerNorm(embed_dims[-1])
        self.head = (
            nn.Linear(embed_dims[-1], num_classes)
            if num_classes > 0
            else torch.nn.Identity()
        )
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dims[-1], 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

    def set_layer_lr_decay(self, layer_lr_decay: float):
        decay_rate = layer_lr_decay
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]

        def _set_lr_scale(m, scale):
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(lambda x: _set_lr_scale(x, lr_scales[i - 1]))
        assert i == depth
        for m in {self.norm_head, self.head}:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))
        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            for p in m.parameters():
                assert hasattr(p, "lr_scale"), p.param_name

        self.apply(_check_lr_scale)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"attention_biases"}

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.layers[0](x)
        start_i = 1
        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
        batch, _, channel = x.shape
        x = x.view(
            batch,
            self.patches_resolution[0] // 4,
            self.patches_resolution[1] // 4,
            channel,
        )
        x = x.permute(0, 3, 1, 2)
        return self.neck(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    def set_imgsz(self, imgsz: list[int] = [1024, 1024]):
        imgsz = [s // 4 for s in imgsz]
        self.patches_resolution = imgsz
        for i, layer in enumerate(self.layers):
            input_resolution = (
                imgsz[0] // 2 ** (i - 1 if i == 3 else i),
                imgsz[1] // 2 ** (i - 1 if i == 3 else i),
            )
            layer.input_resolution = input_resolution
            if layer.downsample is not None:
                layer.downsample.input_resolution = input_resolution
            if isinstance(layer, BasicLayer):
                for b in layer.blocks:
                    b.input_resolution = input_resolution
