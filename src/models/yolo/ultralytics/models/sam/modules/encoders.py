from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import LayerNorm2d
from .blocks import (
    Block,
    CXBlock,
    Fuser,
    MaskDownSampler,
    MultiScaleBlock,
    PatchEmbed,
    PositionEmbeddingRandom,
    PositionEmbeddingSine,
)


class ImageEncoderViT(nn.Module):

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.pos_embed: nn.Parameter | None = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            )
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = (
                F.interpolate(
                    self.pos_embed.permute(0, 3, 1, 2),
                    scale_factor=self.img_size / 1024,
                ).permute(0, 2, 3, 1)
                if self.img_size != 1024
                else self.pos_embed
            )
            x = x + pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.neck(x.permute(0, 3, 1, 2))


class PromptEncoder(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: tuple[int, int],
        input_image_size: tuple[int, int],
        mask_in_chans: int,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.num_point_embeddings: int = 4
        point_embeddings = [
            nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self, points: torch.Tensor, labels: torch.Tensor, pad: bool
    ) -> torch.Tensor:
        points = points + 0.5
        if pad:
            padding_point = torch.zeros(
                (points.shape[0], 1, 2), dtype=points.dtype, device=points.device
            )
            padding_label = -torch.ones(
                (labels.shape[0], 1), dtype=labels.dtype, device=labels.device
            )
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        point_embedding[labels == 2] += self.point_embeddings[2].weight
        point_embedding[labels == 3] += self.point_embeddings[3].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        return self.mask_downscaling(masks)

    @staticmethod
    def _get_batch_size(
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> int:
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def forward(
        self,
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim),
            dtype=self.point_embeddings[0].weight.dtype,
            device=self.point_embeddings[0].weight.device,
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=boxes is None)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        return (sparse_embeddings, dense_embeddings)


class MemoryEncoder(nn.Module):

    def __init__(
        self, out_dim, in_dim=256, interpol_size: tuple[int, int] | None = None
    ):
        super().__init__()
        self.mask_downsampler = MaskDownSampler(
            kernel_size=3, stride=2, padding=1, interpol_size=interpol_size
        )
        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = Fuser(CXBlock(dim=256), num_layers=2)
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=64)
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> dict:
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)
        pix_feat = pix_feat.to(masks.device)
        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)
        pos = self.position_encoding(x).to(x.dtype)
        return {"vision_features": x, "vision_pos_enc": [pos]}


class ImageEncoder(nn.Module):

    def __init__(self, trunk: nn.Module, neck: nn.Module, scalp: int = 0):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        ), f"Channel dims of trunk {self.trunk.channel_list} and neck {self.neck.backbone_channel_list} do not match."

    def forward(self, sample: torch.Tensor):
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            features, pos = (features[: -self.scalp], pos[: -self.scalp])
        src = features[-1]
        return {"vision_features": src, "vision_pos_enc": pos, "backbone_fpn": features}


class FpnNeck(nn.Module):

    def __init__(
        self,
        d_model: int,
        backbone_channel_list: list[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: list[int] | None = None,
    ):
        super().__init__()
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=256)
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )
            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in {"sum", "avg"}
        self.fuse_type = fuse_type
        if fpn_top_down_levels is None:
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: list[torch.Tensor]):
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        prev_features = None
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=x.dtype),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=None if self.fpn_interp_model == "nearest" else False,
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)
        return (out, pos)


class Hiera(nn.Module):

    def __init__(
        self,
        embed_dim: int = 96,
        num_heads: int = 1,
        drop_path_rate: float = 0.0,
        q_pool: int = 3,
        q_stride: tuple[int, int] = (2, 2),
        stages: tuple[int, ...] = (2, 3, 16, 3),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        window_pos_embed_bkg_spatial_size: tuple[int, int] = (14, 14),
        window_spec: tuple[int, ...] = (8, 4, 14, 7),
        global_att_blocks: tuple[int, ...] = (12, 16, 20),
        return_interm_layers=True,
    ):
        super().__init__()
        assert len(stages) == len(window_spec)
        self.window_spec = window_spec
        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers
        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3)
        )
        self.global_att_blocks = global_att_blocks
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        cur_stage = 1
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dim_out = embed_dim
            window_size = self.window_spec[cur_stage - 1]
            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size
            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )
            embed_dim = dim_out
            self.blocks.append(block)
        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for (x, y) in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self._get_pos_embed(x.shape[1:3])
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == self.stage_ends[-1] or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)
        return outputs
