"""Build a lesion-ROI-cropped variant of BUS-BRA, from its own shipped masks.

    python -m scripts.build_bus_bra_roi

Why this exists: `README.md` already names lesion-ROI cropping as probably worth more
than any backbone or objective swap tried so far — BINDS trains on lesion crops, not
whole frames, and attributes much of its reported performance to that. This project's
ultrasound pipeline never cropped to the lesion; mammography does (`crop_to_tissue`,
`prepare_mg_crops.py`), ultrasound didn't, for no principled reason.

**The mask is read, used to compute one crop rectangle, and discarded.** It is never
saved, never loaded as a model input, never a training target. `build_bus_bra.py`
already refuses to extract `Masks/` at all, for exactly the reason this script has to
restate: training on a mask leaks the label through its blob shape. A crop rectangle
carries far less information than the mask's outline — it says roughly "the lesion is
around here", not "the lesion has exactly this shape" — which is the same category of
signal a radiologist's own framing choice already gives away (they centre the probe on
what they found), not a new leak this script introduces.

**Non-destructive, unlike `prepare_mg_crops.py`.** That script overwrites mammograms in
place because the tissue bbox is unambiguous and always safe. A lesion bbox is a bigger,
more consequential change — this project's whole method is never conflating two things
that moved at once (see every RESULT_*.md), so full-frame and lesion-cropped US have to
stay two arms, not one arm silently replacing the other. Cropped images go to a new
`images_roi/` directory; `manifests/bus_bra_roi.csv` is a new manifest, byte-identical to
`bus_bra.csv` except `image_path`. Nothing under `bus_bra.csv` or `data/bus_bra/images/`
is touched.

Requires `data/bus_bra/BUSBRA.zip` to already be on disk (`build_bus_bra.py` fetches it).
Re-running is idempotent: images are re-cropped from the same source, in place under
`images_roi/`.
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

# Standalone copy of bjepa.transforms.crop_to_lesion's logic. Not imported from there
# because bjepa.transforms pulls in torch at module scope for its other (training-time)
# functions, which this build-time script has no other reason to require. Keep the two in
# sync by hand if crop_to_lesion's constants or bbox logic ever change; the real trained
# code always goes through bjepa.transforms, this is only for producing the manifest.
LESION_MARGIN = 0.15
LESION_MIN_KEEP_AREA = 0.03
LESION_MAX_KEEP_AREA = 0.95


def crop_to_lesion(img: Image.Image, mask: Image.Image, margin: float = LESION_MARGIN,
                   square: bool = False) -> Image.Image:
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.NEAREST)
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
    if square:
        # Aspect-preserving variant: a square box centred on the lesion, side = the longer
        # box edge plus margin, shifted (not shrunk) to stay inside the image where possible.
        # Resizing a square crop to 224x224 does not stretch the texture, so the lesion's
        # height/width ratio is not encoded as anisotropic speckle.
        side = int(round(max(bh, bw) * (1 + 2 * margin)))
        side = min(side, H, W)
        cy, cx = (top + bottom) / 2, (left + right) / 2
        top = int(round(min(max(cy - side / 2, 0), H - side)))
        left = int(round(min(max(cx - side / 2, 0), W - side)))
        bottom, right = top + side - 1, left + side - 1
        return img.crop((left, top, right + 1, bottom + 1))
    my, mx = int(round(bh * margin)), int(round(bw * margin))
    top, bottom = max(0, top - my), min(H - 1, bottom + my)
    left, right = max(0, left - mx), min(W - 1, right + mx)
    return img.crop((left, top, right + 1, bottom + 1))

ARCHIVE = os.path.join("data", "bus_bra", "BUSBRA.zip")
OUT_DIR = os.path.join("data", "bus_bra", "images_roi")
SRC_MANIFEST = os.path.join("manifests", "bus_bra.csv")
OUT_MANIFEST = os.path.join("manifests", "bus_bra_roi.csv")


def _mask_name(image_path: str) -> str:
    # data/bus_bra/images/bus_0001-l.png -> BUSBRA/Masks/mask_0001-l.png
    base = os.path.basename(image_path)                # bus_0001-l.png
    return "BUSBRA/Masks/" + base.replace("bus_", "mask_", 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=LESION_MARGIN,
                    help="box margin as a fraction of box height/width (default 0.15)")
    ap.add_argument("--square", action="store_true",
                    help="square, aspect-preserving box (no texture stretch on resize)")
    ap.add_argument("--suffix", default="",
                    help="written to images_roi<suffix>/ and bus_bra_roi<suffix>.csv; "
                         "empty keeps the original paths")
    args = ap.parse_args()
    out_dir, out_manifest = OUT_DIR + args.suffix, OUT_MANIFEST.replace(".csv", args.suffix + ".csv")
    print(f"margin {args.margin}  square {args.square}  ->  {out_dir}, {out_manifest}")
    if not os.path.exists(ARCHIVE):
        raise FileNotFoundError(
            f"{ARCHIVE} not found — run `python -m scripts.build_bus_bra` first "
            f"(it downloads the archive and never deletes it after extracting Images/)."
        )
    if not os.path.exists(SRC_MANIFEST):
        raise FileNotFoundError(f"{SRC_MANIFEST} not found — run build_bus_bra first.")

    df = pd.read_csv(SRC_MANIFEST)
    os.makedirs(out_dir, exist_ok=True)

    z = zipfile.ZipFile(ARCHIVE)
    names = set(z.namelist())

    n, n_cropped, n_full, n_missing_mask = 0, 0, 0, 0
    out_rows = []
    for _, row in df.iterrows():
        src_path = row["image_path"]                    # data/bus_bra/images/bus_XXXX-y.png
        out_path = os.path.join(out_dir, os.path.basename(src_path))
        mname = _mask_name(src_path)

        img = Image.open(src_path).convert("L")
        if mname in names:
            mask = Image.open(io.BytesIO(z.read(mname))).convert("L")
            cropped = crop_to_lesion(img, mask, args.margin, args.square)
        else:
            cropped = img
            n_missing_mask += 1

        if cropped.size != img.size:
            n_cropped += 1
        else:
            n_full += 1
        cropped.save(out_path)

        r = row.to_dict()
        r["image_path"] = out_path
        out_rows.append(r)

        n += 1
        if n % 300 == 0:
            print(f"  [{n}/{len(df)}]")

    pd.DataFrame(out_rows).to_csv(out_manifest, index=False)
    print(f"\nwrote {n} images -> {out_dir}")
    print(f"wrote manifest -> {out_manifest}")
    print(f"cropped to lesion bbox: {n_cropped}  |  left full-frame "
          f"(mask empty/near-full, see LESION_MIN/MAX_KEEP_AREA): {n_full - n_missing_mask}"
          f"  |  no mask found in archive: {n_missing_mask}")


if __name__ == "__main__":
    main()
