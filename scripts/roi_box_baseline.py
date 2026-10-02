"""Does the ROI crop box alone predict malignancy?

The ROI experiment crops each BUS-BRA image to the bounding box of its expert lesion mask
(15% margin) and probes the crop. The box is computed from the mask on every split,
test included. If the box geometry by itself (size, aspect ratio, position) separates
benign from malignant, part of the ROI gain could come from the box rather than from
the pixels inside it.

This fits the same probe recipe (StandardScaler + LogisticRegression, C=1) on box
features only, trained on BUS-BRA train, scored on validation and test. No image
pixels are read.

    python -m scripts.roi_box_baseline
"""
from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ARCHIVE = os.path.join("data", "bus_bra", "BUSBRA.zip")
MANIFEST = os.path.join("manifests", "bus_bra.csv")
FEATURES = ["box_area_frac", "box_h_frac", "box_w_frac", "aspect_hw",
            "center_y", "center_x", "mask_fill", "img_aspect"]


def box_features(img_path: str, z: zipfile.ZipFile) -> dict:
    base = os.path.basename(img_path)
    mask = Image.open(io.BytesIO(z.read("BUSBRA/Masks/" + base.replace("bus_", "mask_", 1)))).convert("L")
    return box_features_from_mask(mask, Image.open(img_path).size)


def box_features_from_mask(mask: Image.Image, img_size: tuple) -> dict:
    """Box geometry of a lesion mask, in the coordinates of an image of size (w, h)."""
    img_w, img_h = img_size
    if mask.size != (img_w, img_h):
        mask = mask.resize((img_w, img_h), Image.NEAREST)
    a = np.asarray(mask) > 0
    ys, xs = np.where(a)
    bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    return {
        "box_area_frac": bh * bw / (img_h * img_w),
        "box_h_frac": bh / img_h,
        "box_w_frac": bw / img_w,
        "aspect_hw": bh / bw,
        "center_y": (ys.min() + ys.max()) / 2 / img_h,
        "center_x": (xs.min() + xs.max()) / 2 / img_w,
        # fraction of the box covered by the lesion: irregular shapes fill less of it
        "mask_fill": a.sum() / (bh * bw),
        "img_aspect": img_h / img_w,
    }


def main() -> None:
    df = pd.read_csv(MANIFEST)
    z = zipfile.ZipFile(ARCHIVE)
    feats = pd.DataFrame([box_features(p, z) for p in df.image_path])
    df = pd.concat([df.reset_index(drop=True), feats], axis=1)

    tr = df[df.split == "train"]
    print(f"train {len(tr)}  validation {sum(df.split == 'validation')}  test {sum(df.split == 'test')}\n")

    def fit_score(cols: list[str], label: str) -> None:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        clf.fit(tr[cols], tr.label)
        out = []
        for split in ("validation", "test"):
            s = df[df.split == split]
            out.append(f"{split} {roc_auc_score(s.label, clf.predict_proba(s[cols])[:, 1]):.3f}")
        print(f"{label:38s} " + "   ".join(out))

    fit_score(FEATURES, "all box features")
    fit_score(["box_area_frac", "box_h_frac", "box_w_frac", "aspect_hw", "center_y", "center_x", "img_aspect"],
              "box only (no mask_fill)")
    for c in FEATURES:
        fit_score([c], f"  {c}")


if __name__ == "__main__":
    main()
