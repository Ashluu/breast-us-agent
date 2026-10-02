"""Image loading + per-modality augmentation.

Follows the BINDS convention (Nat. Biomed. Eng. 2026): mild, anatomy-safe
augmentation only — 90-degree rotations, gentle brightness/contrast, flips,
light center-crop. No elastic warp, blur, random erasing or colour jitter.
Tensors stay in [0, 1]; there is no ImageNet mean/std normalization.
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

# BINDS resizes ultrasound bilinearly and mammography with nearest-neighbour.
# PIL's own default is bicubic, which matches neither.
RESAMPLE = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC}
# `path` = histology patches (TCGA-UT). Bilinear like ultrasound, not nearest: these are
# 256x256 crops resized to 224, a mild downsample where nearest would alias the cell texture
# the encoder is supposed to read. Mammography keeps nearest to follow BINDS.
DEFAULT_RESAMPLE = {"us": "bilinear", "mg": "nearest", "path": "bilinear"}

# Modalities whose colour carries signal. Ultrasound and mammography are physically grayscale,
# so `.convert("L")` costs nothing there. Histology is the opposite: H&E staining IS the
# readout — haematoxylin dyes nuclei blue-purple, eosin dyes cytoplasm pink — and flattening
# that to luminance throws away exactly what the encoder should read. Phikon was pretrained
# on RGB patches too, so graying them also puts the input off its training distribution.
RGB_MODALITIES = {"path"}


TISSUE_THRESHOLD = 10      # 0-255; a full-field mammogram's true background sits near 0
TISSUE_MARGIN = 0.02       # fraction of each dimension kept as padding around the bbox
MIN_KEEP_AREA = 0.40       # below this, distrust the threshold and return the image uncropped


def crop_to_tissue(img: Image.Image) -> Image.Image:
    """Crop a full-field mammogram down to its breast bounding box.

    Full mammograms (VinDr, RSNA, DDSM's `full` rows — none of the three sources
    crop on the way in, see `scripts/build_ddsm.py` / `build_rsna.py` / `build_vindr.py`)
    devote most of the frame to true black background outside the breast. Left
    uncropped, that background eats patch budget the encoder could spend on tissue.
    Doing the crop here rather than in any one `build_*.py` keeps all three sources
    consistent without re-downloading or re-saving anything.

    Threshold-based, not learned — and that cuts both ways. Checked against a live
    DDSM `full` row (`img_00091.png`): a fixed 10/255 cutoff caught only 2.3% of
    pixels (a bright artifact/calcification, not the breast) and would have cropped
    5491x2956 down to 710x777 — discarding almost the whole mammogram. DDSM's mirror
    apparently isn't intensity-normalized the same way our own DICOM conversion
    (`build_vindr.py`) is, so one fixed threshold doesn't transfer cleanly across
    sources. `MIN_KEEP_AREA` catches that case: if the naive bbox would keep less
    than 40% of the frame, distrust it and return the image uncropped rather than
    risk silently training on an artifact crop.
    """
    arr = np.asarray(img)
    mask = arr > TISSUE_THRESHOLD
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    top, bottom = int(ys.min()), int(ys.max())
    left, right = int(xs.min()), int(xs.max())
    h, w = arr.shape
    my, mx = int(round(h * TISSUE_MARGIN)), int(round(w * TISSUE_MARGIN))
    top, bottom = max(0, top - my), min(h - 1, bottom + my)
    left, right = max(0, left - mx), min(w - 1, right + mx)
    if (bottom - top + 1) * (right - left + 1) < MIN_KEEP_AREA * h * w:
        return img
    return img.crop((left, top, right + 1, bottom + 1))



LESION_MARGIN = 0.15       # fraction of the bbox's own size added as padding on each side
LESION_MIN_KEEP_AREA = 0.03   # below this, the mask is probably wrong/empty; leave uncropped
LESION_MAX_KEEP_AREA = 0.95   # bbox this close to the full frame isn't a real ROI; skip


def crop_to_lesion(img: Image.Image, mask: Image.Image) -> Image.Image:
    """Crop an ultrasound frame to its lesion bounding box, from a segmentation mask.

    The mask is used ONLY to compute a crop rectangle — its pixels are never loaded as a
    model input or a training target, and it is discarded immediately after this call.
    That is the whole safety argument: a mask fed to the model leaks the label through its
    blob shape (this is exactly why `build_bus_bra.py` / `clean_ddsm.py` never extract or
    save mask files at all — see their docstrings). A crop RECTANGLE carries far less
    information than the mask's outline, and BINDS's own reported gains from lesion-ROI
    cropping are attributed to framing, not to handing the model the segmentation.

    Margin is a FRACTION OF THE BBOX, not the frame — `TISSUE_MARGIN` in `crop_to_tissue`
    is relative to the full mammogram because that crop is already loose (background vs.
    breast). A lesion bbox is tight by construction, so a frame-relative margin would be a
    near-constant number of pixels regardless of lesion size; a bbox-relative margin keeps
    context proportional to the lesion — a 20px lesion gets a few px of surrounding tissue,
    a 200px one gets a comparably wide margin, both readable the same way.

    Two floors, both there because a wrong bbox is worse than no crop:
    * `LESION_MIN_KEEP_AREA` — an empty or near-empty mask (segmentation failure, or a
      normal/no-finding image with no annotated lesion) would crop to a few pixels or
      raise on an empty bbox. Caught here and returned uncropped instead.
    * `LESION_MAX_KEEP_AREA` — a mask that covers almost the whole frame is not doing any
      framing work; cropping to it is a no-op with extra CPU, not a bug, but flagged the
      same way as a distrust signal in the caller's summary.
    """
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.NEAREST)   # masks are label maps, never blur them
    arr = np.asarray(mask)
    ys, xs = np.where(arr > 0)
    if len(ys) == 0:
        return img
    top, bottom = int(ys.min()), int(ys.max())
    left, right = int(xs.min()), int(xs.max())
    bh, bw = bottom - top + 1, right - left + 1
    W, H = img.size
    if bh * bw < LESION_MIN_KEEP_AREA * H * W:
        return img
    if bh * bw > LESION_MAX_KEEP_AREA * H * W:
        return img
    my, mx = int(round(bh * LESION_MARGIN)), int(round(bw * LESION_MARGIN))
    top, bottom = max(0, top - my), min(H - 1, bottom + my)
    left, right = max(0, left - mx), min(W - 1, right + mx)
    return img.crop((left, top, right + 1, bottom + 1))


def load_image(
    path: str,
    size: Tuple[int, int] = (224, 224),
    resample: str = "bilinear",
    modality: str = "",
) -> Tensor:
    """Load an image, resize, scale to [0,1]. Returns ``[1, H, W]`` — or ``[3, H, W]`` for
    the modalities in ``RGB_MODALITIES``, where colour is signal rather than decoration.

    Mammograms are cropped to tissue **offline**, once, by `scripts/prepare_mg_crops.py`
    — not here. Doing it per-`__getitem__` recomputed the same deterministic bbox on
    every epoch, every sample, for the life of training; baking it into the PNG on disk
    (once, on the machine that already holds the raw images — no re-download needed)
    removes that redundant CPU work and shrinks the files besides.
    """
    if resample not in RESAMPLE:
        raise ValueError(f"resample must be one of {list(RESAMPLE)}, got '{resample}'")
    mode = "RGB" if modality in RGB_MODALITIES else "L"
    img = Image.open(path).convert(mode).resize(size[::-1], RESAMPLE[resample])  # PIL wants (W,H)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if mode == "RGB":
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()      # [3, H, W]
    return torch.from_numpy(arr)[None, ...]                             # [1, H, W]


def to_3ch(x: Tensor) -> Tensor:
    """Grayscale -> 3 channels, for ImageNet/DINOv2 pretrained backbones."""
    return x.repeat(3, *([1] * (x.dim() - 1))) if x.shape[0] == 1 else x


def brightness_contrast(x: Tensor, lo: float = 0.5, hi: float = 1.5) -> Tensor:
    x = x * random.uniform(lo, hi)                       # brightness
    mean = x.mean()
    x = (x - mean) * random.uniform(lo, hi) + mean       # contrast
    return x


def _center_crop_resize(x: Tensor, ratio: float) -> Tensor:
    H, W = x.shape[-2], x.shape[-1]
    ch, cw = max(1, int(round(H * ratio))), max(1, int(round(W * ratio)))
    top, left = (H - ch) // 2, (W - cw) // 2
    x = x[..., top : top + ch, left : left + cw]
    x4 = x.reshape(-1, 1, x.shape[-2], x.shape[-1]).float()
    x4 = F.interpolate(x4, size=(H, W), mode="bilinear", align_corners=False)
    return x4.reshape(x.shape[0], H, W)


def lesion_aug(x: Tensor) -> Tensor:
    """BINDS augmentation for 2D modalities.

    One departure from BINDS: **quarter-turns are only applied to square inputs.**
    BINDS rotates by any multiple of 90 degrees, which is safe on the square-ish
    lesion crops it trains on. It is wrong twice on a portrait mammogram (280x210):
    it transposes the tensor, so the batch will not even stack — and, worse than
    the crash, a 90-degree-rotated mammogram is an image no mammography unit can
    produce. The breast has an orientation; laterality and view (CC/MLO) are part
    of the signal. So non-square inputs get half-turns only, which preserve both
    the shape and the anatomy.
    """
    x = brightness_contrast(x, 0.5, 1.5)
    if random.random() < 0.5:
        x = torch.flip(x, dims=(-1,))                    # horizontal flip: left/right breast
    square = x.shape[-1] == x.shape[-2]
    k = random.randint(0, 3) if square else random.choice((0, 2))
    if k:
        x = torch.rot90(x, k, dims=(-2, -1))
    if random.random() < 0.5:
        x = _center_crop_resize(x, random.uniform(0.8, 1.0))
    return x.clamp(0.0, 1.0)


def augment_batch(x: Tensor) -> Tensor:
    """Augment ``[B, C, H, W]`` with fresh randomness per sample.

    Applying the aug to the whole batch at once would share one brightness and
    rotation draw across every image in it, which is not what BINDS does.
    """
    return torch.stack([lesion_aug(sample) for sample in x])


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    base = torch.rand(3, 32, 32)
    out = augment_batch(base.unsqueeze(0).repeat(6, 1, 1, 1))
    pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    same = sum(torch.allclose(out[i], out[j]) for i, j in pairs)
    assert same == 0, "augmentation must draw per-sample randomness"
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0
    print(f"shape={tuple(out.shape)} identical_pairs={same}/{len(pairs)} range=[0,1]")
    print("OK")
