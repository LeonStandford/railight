from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ContextBranch",
    "IAAlign",
    "OAAlign",
    "PatchDomainDiscriminator",
    "CategoryPrototypes",
    "grad_reverse",
]


class _GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, weight: float) -> torch.Tensor:
        ctx.weight = float(weight)
        return features.view_as(features)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (-ctx.weight * grad_output, None)


def grad_reverse(features: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(features, weight)


def _norm(channels: int) -> nn.Module:
    return nn.GroupNorm(min(32, channels), channels)


def _conv_norm_act(in_ch: int, out_ch: int, k: int = 1, act: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
        _norm(out_ch),
    ]
    if act:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class ContextBranch(nn.Module):

    def __init__(
        self,
        channels: int,
        impl: str = "pool",
        max_tokens: int = 1024,
        mamba_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.max_tokens = max(int(max_tokens), 16)
        self.impl = str(impl or "pool").lower()
        self.mamba = None
        if self.impl == "mamba":
            try:
                from mamba_ssm import Mamba

                self.mamba = Mamba(d_model=channels, **(mamba_kwargs or {}))
            except Exception as exc:
                print(
                    f"[da-align] mamba_ssm unavailable ({exc}); "
                    f"falling back to the pooled context branch."
                )
                self.impl = "pool"
        if self.impl != "mamba":
            hidden = max(channels // 4, 8)
            self.local = nn.Conv2d(
                channels, channels, 7, padding=3, groups=channels, bias=False
            )
            self.norm = _norm(channels)
            self.act = nn.SiLU(inplace=True)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, channels, 1),
                nn.Sigmoid(),
            )

    def _mamba_forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        stride = 1
        while (h // stride) * (w // stride) > self.max_tokens:
            stride *= 2
        reduced = x if stride == 1 else F.avg_pool2d(x, stride)
        rh, rw = reduced.shape[-2:]
        seq = reduced.flatten(2).transpose(1, 2)
        seq = self.mamba(seq)
        out = seq.transpose(1, 2).reshape(b, c, rh, rw)
        if stride != 1:
            out = F.interpolate(out, size=(h, w), mode="nearest")
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.impl == "mamba":
            return self._mamba_forward(x)
        y = self.act(self.norm(self.local(x)))
        return y * self.gate(y)


class _DualBranch(nn.Module):

    def __init__(
        self,
        channels: int,
        reduction: float = 2.0,
        context_impl: str = "pool",
        max_tokens: int = 1024,
    ) -> None:
        super().__init__()
        c_red = max(1, int(round(channels / max(reduction, 1e-6))))
        self.encode_seq = _conv_norm_act(2 * channels, c_red, k=1)
        self.encode_conv = _conv_norm_act(2 * channels, c_red, k=1)
        self.context = ContextBranch(c_red, impl=context_impl, max_tokens=max_tokens)
        self.conv_pipe = _conv_norm_act(c_red, c_red, k=3)
        self.fuse = _conv_norm_act(2 * c_red, channels, k=1, act=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def refine(self, paired: torch.Tensor) -> torch.Tensor:
        z_seq = self.context(self.encode_seq(paired))
        z_conv = self.conv_pipe(self.encode_conv(paired))
        return self.fuse(torch.cat([z_seq, z_conv], dim=1))


class IAAlign(_DualBranch):

    def __init__(
        self,
        channels: int,
        reduction: float = 2.0,
        context_impl: str = "pool",
        max_tokens: int = 1024,
    ) -> None:
        super().__init__(channels, reduction, context_impl, max_tokens)
        self.channels = channels
        self.visual_prompt = nn.Parameter(torch.zeros(channels))
        nn.init.normal_(self.visual_prompt, std=0.02)

    def forward(self, f_in: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f_in.shape
        prompt = self.visual_prompt.view(1, c, 1, 1).expand(b, c, h, w)
        return f_in + self.gate * self.refine(torch.cat([f_in, prompt], dim=1))


class OAAlign(_DualBranch):

    def __init__(
        self,
        channels: int,
        num_classes: int,
        prototype_dim: int,
        reduction: float = 2.0,
        context_impl: str = "pool",
        max_tokens: int = 1024,
    ) -> None:
        super().__init__(channels, reduction, context_impl, max_tokens)
        self.channels = channels
        self.num_classes = num_classes
        self.cat_proj = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.proto_proj = nn.Linear(prototype_dim, channels)

    def forward(
        self,
        f_in: torch.Tensor,
        prototypes: torch.Tensor,
        present: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.cat_proj(f_in)
        if present is not None:
            mask = present.to(logits.dtype).view(logits.shape[0], -1, 1, 1)
            empty = mask.sum(dim=1, keepdim=True) <= 0
            mask = torch.where(empty, torch.ones_like(mask), mask)
            logits = logits.masked_fill(mask <= 0, float("-inf"))
        weights = F.softmax(logits, dim=1).permute(0, 2, 3, 1)
        embedded = self.proto_proj(prototypes.to(f_in.dtype))
        object_vec = torch.einsum("bhwk,kc->bhwc", weights, embedded)
        object_vec = object_vec.permute(0, 3, 1, 2).contiguous()
        return f_in + self.gate * self.refine(torch.cat([f_in, object_vec], dim=1))


class PatchDomainDiscriminator(nn.Module):

    def __init__(self, in_channels: int, hidden: int = 256, max_size: int = 80) -> None:
        super().__init__()
        width = max(int(hidden), 1)
        self.max_size = max(int(max_size), 0)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 1, 1),
        )

    def forward(self, x: torch.Tensor, grl_lambda: float = 1.0) -> torch.Tensor:
        if self.max_size:
            h, w = x.shape[-2:]
            if h > self.max_size or w > self.max_size:
                x = F.adaptive_avg_pool2d(
                    x, (min(h, self.max_size), min(w, self.max_size))
                )
        return self.net(grad_reverse(x, grl_lambda))


def _clip_text_embeddings(
    class_names: Sequence[str],
    model_name: str,
    template: str = "a photo of a {}",
) -> Optional[torch.Tensor]:
    try:
        from transformers import CLIPModel, CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).eval()
        prompts = [template.format(name.replace("_", " ")) for name in class_names]
        batch = tokenizer(prompts, padding=True, return_tensors="pt")
        with torch.no_grad():
            output = model.get_text_features(**batch)
            feats = (
                output if torch.is_tensor(output)
                else getattr(output, "text_embeds", None)
            )
            if feats is None:
                feats = output.pooler_output
                projection = getattr(model, "text_projection", None)
                if projection is not None:
                    feats = projection(feats)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats.detach().float().cpu()
    except Exception as exc:
        print(f"[da-align] CLIP prototypes unavailable ({exc}); using random prototypes.")
        return None


class CategoryPrototypes(nn.Module):

    def __init__(
        self,
        class_names: Sequence[str],
        embed_dim: int = 512,
        source: str = "clip",
        clip_model: str = "openai/clip-vit-base-patch32",
        learnable: bool = False,
    ) -> None:
        super().__init__()
        self.class_names = list(class_names)
        embeddings = (
            _clip_text_embeddings(self.class_names, clip_model)
            if str(source).lower() == "clip"
            else None
        )
        self.from_clip = embeddings is not None
        if embeddings is None:
            generator = torch.Generator().manual_seed(0)
            embeddings = torch.randn(
                len(self.class_names), embed_dim, generator=generator
            )
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        embeddings = embeddings.detach()
        self.embed_dim = int(embeddings.shape[-1])
        if learnable:
            self.embeddings = nn.Parameter(embeddings)
        else:
            self.register_buffer("embeddings", embeddings)

    def forward(self) -> torch.Tensor:
        return self.embeddings
