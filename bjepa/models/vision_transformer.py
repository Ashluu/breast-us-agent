"""The I-JEPA encoder: a plain ViT that can run on a *subset* of patch tokens.

I-JEPA needs three things a stock ViT wrapper does not give you:

1. **Patch tokens only.** No CLS token — I-JEPA's loss lives in patch-latent
   space, and a CLS token would attend across the context/target boundary we are
   trying to enforce. We keep the pretrained positional embedding but drop its
   prefix (CLS/register) entries.
2. **Masked forward.** The context encoder sees only the context patches; passing
   the full image and zeroing the rest would leak target content through
   attention. So we gather the kept tokens *before* the blocks run.
3. **Weights from anywhere.** The init is the whole experiment (arms C vs D), so
   the encoder loads from either timm (`vit_base_patch16_224...`, ImageNet) or a
   HuggingFace DINOv2 checkpoint (`hf:microsoft/rad-dino`, chest X-ray). Both are
   plain ViTs underneath; only the plumbing to reach the blocks differs.

Whatever the source, only the ViT *body* survives — patch embedding, positional
embedding, transformer blocks, final norm. Pretraining heads (MAE decoder, DINO
projection head) are dropped on the floor.
"""

from __future__ import annotations

from typing import Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

HF_PREFIX = "hf:"


def apply_masks(x: Tensor, idx: Tensor) -> Tensor:
    """Gather a token subset. ``x``: [B, N, D], ``idx``: [B, K] -> [B, K, D]."""
    return torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def resize_pos_embed(pos: Tensor, grid: Tuple[int, int]) -> Tensor:
    """Bicubic-resize a [1, N, D] patch positional embedding onto ``grid``."""
    n, d = pos.shape[1], pos.shape[2]
    src = int(round(n ** 0.5))
    if (src, src) == tuple(grid):
        return pos.clone()
    p = pos.reshape(1, src, src, d).permute(0, 3, 1, 2)
    p = F.interpolate(p, size=grid, mode="bicubic", align_corners=False)
    return p.permute(0, 2, 3, 1).reshape(1, grid[0] * grid[1], d).contiguous()


class _ConvPatchEmbed(nn.Module):
    """`transformers` keeps patch embedding as a bare Conv2d; timm returns tokens."""

    def __init__(self, proj: nn.Conv2d):
        super().__init__()
        self.proj = proj

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)      # [B, N, D]


class _HFBlock(nn.Module):
    """A `transformers` DINOv2 layer returns a tuple; timm blocks return a tensor.

    Which one depends on the installed `transformers` version — some releases return
    the hidden_states tensor directly instead of `(hidden_states,)`. Blindly indexing
    `[0]` into a plain tensor silently grabs batch element 0 instead of unpacking the
    tuple, dropping the batch dim without raising — every downstream shape check then
    fails somewhere else (e.g. apply_masks' gather), far from the real cause.
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x: Tensor) -> Tensor:
        out = self.layer(x)
        return out[0] if isinstance(out, tuple) else out


class ViTEncoder(nn.Module):
    """Plain ViT body, forwardable on a subset of patch tokens.

    ``forward(x, idx)`` embeds every patch, adds position, keeps only the tokens
    named by ``idx``, and runs the blocks on those. ``idx=None`` runs the full
    grid — which is what the EMA target encoder does.
    """

    def __init__(
        self,
        name: str = "vit_base_patch16_224.augreg_in21k_ft_in1k",
        pretrained: bool = True,
        img_size=224,
        in_chans: int = 3,
    ):
        super().__init__()
        self.name = name
        # Non-square input is not a nicety here. Mammograms are portrait (~4:3), and
        # AEGIS/BAC-JEPA train at 256x192 / 512x384 / 2048x1536 — never square.
        # Forcing a mammogram into 224x224 stretches the breast, which is a distortion
        # the encoder then has to model. `img_size` may be an int or (H, W).
        h, w = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        self.img_size = (h, w)

        if name.startswith(HF_PREFIX):
            from transformers import AutoModel

            model = AutoModel.from_pretrained(name[len(HF_PREFIX):])
            cfg = model.config
            p = cfg.patch_size
            if in_chans != 3:
                raise ValueError("HF DINOv2 checkpoints expect 3-channel input")
            if h % p or w % p:
                raise ValueError(
                    f"image_size {(h, w)} is not divisible by patch {p}. "
                    f"RAD-DINO is patch-14, so 512x384 does NOT tile — use multiples of 14 "
                    f"(e.g. 280x210, then 560x420: both exactly 4:3)."
                )

            self.embed_dim = cfg.hidden_size
            self.grid_size = (h // p, w // p)
            self.patch_embed = _ConvPatchEmbed(model.embeddings.patch_embeddings.projection)
            self.blocks = nn.ModuleList(_HFBlock(l) for l in model.encoder.layer)
            self.norm = model.layernorm
            # [:, 1:] drops CLS. RAD-DINO is pretrained at 518px; we run 224.
            self.pos_embed = nn.Parameter(
                resize_pos_embed(model.embeddings.position_embeddings[:, 1:], self.grid_size)
            )
        else:
            vit = timm.create_model(
                name, pretrained=pretrained, img_size=(h, w),
                in_chans=in_chans, num_classes=0,
            )
            # timm floors a non-divisible size instead of complaining: ask for 280x210
            # at patch 16 and you get a 17x13 grid covering 272x208, with the last 8
            # rows of every image silently discarded. Fail here instead.
            p = vit.patch_embed.patch_size[0]
            if h % p or w % p:
                raise ValueError(
                    f"image_size {(h, w)} is not divisible by patch {p} — timm would "
                    f"silently crop to {(h // p * p, w // p * p)}. Pick multiples of {p}."
                )
            self.embed_dim = vit.embed_dim
            self.grid_size = tuple(vit.patch_embed.grid_size)
            self.patch_embed = vit.patch_embed
            self.blocks = vit.blocks
            self.norm = vit.norm
            self.pos_embed = nn.Parameter(vit.pos_embed[:, vit.num_prefix_tokens:].clone())

        self.num_patches = self.grid_size[0] * self.grid_size[1]
        if self.pos_embed.shape[1] != self.num_patches:
            raise RuntimeError(
                f"pos_embed has {self.pos_embed.shape[1]} tokens, grid wants {self.num_patches}"
            )

    def forward(self, x: Tensor, idx: Optional[Tensor] = None,
                layer: Optional[int] = None) -> Tensor:
        """`layer` stops after that many blocks instead of running all of them.

        🔴 WHY THIS EXISTS, AND WHY IT IS NOT CHEATING.

        Every probe in this project reads the LAST block. In the MAE / JEPA literature the
        best linear-probe features frequently sit around two thirds of the depth: the final
        blocks specialise to the pretext task — reconstructing or predicting masked
        content — and that specialisation is not what a malignancy head wants. Reading an
        intermediate block costs no training at all, so "which layer" is a question we have
        simply never asked.

        🔴 It is also the easiest place in this repo to manufacture a result. Twelve layers
        x two datasets is twenty-four numbers, and reporting the best of them is how a null
        becomes a finding: with a seed sd near 0.01, the maximum of many draws sits well
        above the truth by chance. So the rule is fixed before any sweep runs: the LAST
        layer stays the headline, the sweep is reported as a CURVE with every point shown,
        and a layer only replaces the default if it is chosen on one dataset and then holds
        on a different one.

        `layer=None` (the default) is the unchanged behaviour, so no existing number moves.
        Negative indices count from the end, and the final norm is applied either way —
        without it the scale differs between layers and the standardiser hides it.
        """
        x = self.patch_embed(x) + self.pos_embed
        if idx is not None:
            x = apply_masks(x, idx)                 # gather BEFORE attention
        blocks = self.blocks
        if layer is not None:
            n = len(self.blocks)
            k = layer + n if layer < 0 else layer
            if not 1 <= k <= n:
                raise ValueError(f"layer must be in [1, {n}] or [-{n}, -1], got {layer}")
            blocks = self.blocks[:k]
        for blk in blocks:
            x = blk(x)
        return self.norm(x)


def load_mae_checkpoint(enc: ViTEncoder, path: str) -> None:
    """Load an MAE-pretrained ViT (USF-MAE) into `enc`, in place.

    USF-MAE is published as a raw `.pth` on Google Drive rather than through timm or
    the HF hub, so there is no `create_model` path to it — hence this. Inspected, it
    is `ViT-B/16 @224`, 3-channel, 12 blocks, width 768: the *same* architecture as
    the ImageNet ViT-B/16 arm, which makes this a pure weight swap and the two arms
    exactly comparable.

    Three things are dropped on the way in, and each one matters:

    * **The MAE decoder** (8 blocks, width 512). It exists to reconstruct pixels. We
      predict latents; it has no job here.
    * **The CLS token and its position.** I-JEPA is patch-only — a CLS token would
      attend across the context/target boundary the method depends on. The checkpoint
      carries 197 positions; we keep the 196 patch ones.
    * **`mask_token`**, which is MAE's, not ours.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)

    # 🔴 Two different kinds of file arrive here and they are not interchangeable.
    #
    #   * A foundation-model checkpoint (USF-MAE, RAD-DINO): a flat state_dict, possibly
    #     under "model" or "state_dict", carrying an MAE decoder and a CLS token to strip.
    #   * **One of OUR OWN P1 checkpoints**, which is a payload dict — "encoder",
    #     "predictor", "optimizer", "cfg", ... — where the encoder weights live under
    #     "encoder" and are ALREADY in this class's own layout.
    #
    # Arm 2 needs the second: it freezes what arm 1 produced. Without this branch it fails
    # with "has no pos_embed", because the payload's top level has no tensors at all.
    if isinstance(ck, dict) and "encoder" in ck and isinstance(ck["encoder"], dict) \
            and any(k.startswith("blocks.") for k in ck["encoder"]):
        run = ck.get("run", {})
        enc.load_state_dict(ck["encoder"], strict=True)
        print(f"  loaded OUR OWN encoder from {path} "
              f"({run.get('epochs_done', '?')} epochs, pool {run.get('pool', '?')}"
              f"{', 🔴 SMOKE' if run.get('smoke') else ''})")
        if run.get("smoke"):
            raise ValueError(
                f"{path} is a SMOKE checkpoint (a --limit run kept for provenance). "
                f"Freezing it would measure a model trained on a handful of images.")
        return

    sd = ck.get("model", ck.get("state_dict", ck))

    tgt = {"patch_embed": enc.patch_embed, "blocks": enc.blocks, "norm": enc.norm}
    own, loaded = {}, 0
    for k, v in sd.items():
        head = k.split(".")[0]
        if head in tgt:
            own[k] = v
            loaded += 1

    missing = []
    for name, mod in tgt.items():
        sub = {k[len(name) + 1:]: v for k, v in own.items() if k.startswith(name + ".")}
        got = mod.load_state_dict(sub, strict=False)
        missing += [f"{name}.{m}" for m in got.missing_keys]

    pos = sd.get("pos_embed")
    if pos is None:
        raise KeyError(f"{path} has no pos_embed")
    n_prefix = pos.shape[1] - enc.num_patches          # 197 - 196 = 1 (the CLS slot)
    if n_prefix < 0:
        pos = resize_pos_embed(pos[:, 1:], enc.grid_size)
    else:
        pos = pos[:, n_prefix:]
    with torch.no_grad():
        enc.pos_embed.copy_(pos)

    print(f"  loaded {loaded} tensors from {path} (MAE decoder + cls_token dropped)")
    if missing:
        raise RuntimeError(
            f"{len(missing)} encoder tensors were NOT in the checkpoint, e.g. {missing[:4]}. "
            f"The architecture does not match — refusing to train on a half-initialised encoder."
        )


def build_encoder(cfg: dict) -> ViTEncoder:
    m = cfg["model"]
    enc = ViTEncoder(
        name=m["encoder"],
        pretrained=bool(m.get("pretrained", True)),
        img_size=tuple(cfg.get("image_size", [224, 224])),
    )
    if m.get("init_checkpoint"):
        load_mae_checkpoint(enc, m["init_checkpoint"])
    return enc


if __name__ == "__main__":
    for size in [224, (288, 208)]:                      # square, and portrait (both /16)
        enc = ViTEncoder(pretrained=False, img_size=size)
        h, w = enc.img_size
        x = torch.randn(2, 3, h, w)
        full = enc(x)
        idx = torch.stack([torch.randperm(enc.num_patches)[:40] for _ in range(2)])
        part = enc(x, idx)
        print(f"img={enc.img_size} grid={enc.grid_size} ({enc.num_patches} tokens) "
              f"dim={enc.embed_dim} full={tuple(full.shape)} masked={tuple(part.shape)}")
        assert full.shape == (2, enc.num_patches, enc.embed_dim)
        assert part.shape == (2, 40, enc.embed_dim)
    print("OK")
