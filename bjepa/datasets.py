"""One dataset class for every 2D modality.

BUSI and CBIS-DDSM manifests share the same three columns, so there is no reason
for two classes:

    image_path,risk[,split]

`risk` is the binary benign(0)-vs-malignant(1) target. Rows whose `risk` is
IGNORE_INDEX (BUSI's `normal` class) are dropped up front — the loss would skip
them anyway, and dropping them keeps the reported dataset size honest.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import DEFAULT_RESAMPLE, load_image, to_3ch

IGNORE_INDEX = -100

# CBIS-DDSM's mirror mixes lesion masks in with the mammograms (44% of rows!), and a
# mask is a black canvas whose blob shape correlates with malignancy — free label
# leakage. `scripts/clean_ddsm.py` tags them; this is the guard that makes forgetting
# to filter them impossible rather than merely inadvisable.
DROP_IMAGE_TYPES = ("mask",)


def _drop_masks(df: pd.DataFrame, manifest: str) -> pd.DataFrame:
    if "image_type" not in df.columns:
        return df
    bad = df["image_type"].isin(DROP_IMAGE_TYPES)
    if bad.any():
        print(f"  {os.path.basename(manifest):24s} dropping {int(bad.sum())} "
              f"{'/'.join(DROP_IMAGE_TYPES)} rows (not mammograms)")
    return df[~bad]


def label_subset(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Keep `frac` of the LABELLED training rows, for label-efficiency curves.

    Sampled by patient, not by image: two views of one lesion (BUS-BRA `-l`/`-r`) either
    both stay or both go, so a 5% subset is not secretly a 2.5%-of-patients subset with
    duplicates. Stratified by `risk`, so 5% of 1,114 images still has both classes (at least
    two patients each). Seeded: the same (frac, seed) is the same subset for every encoder,
    which is what makes the curves comparable — the probe and the supervised run see the
    identical labelled images.
    """
    if frac >= 1.0:
        return df
    rng = np.random.default_rng(int(seed))
    unit = "patient_id" if "patient_id" in df.columns else None
    parts = []
    for _, g in df.groupby("risk", sort=True):
        units = g[unit].unique() if unit else g.index.to_numpy()
        n = min(len(units), max(2, int(round(frac * len(units)))))
        pick = rng.choice(units, size=n, replace=False)
        parts.append(g[g[unit].isin(pick)] if unit else g.loc[pick])
    return pd.concat(parts).reset_index(drop=True)


class LesionDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        modality: str,
        split: str = "",
        image_size: Tuple[int, int] = (224, 224),
        root: str = "",
        resample: Optional[str] = None,
    ):
        if not os.path.exists(manifest):
            raise FileNotFoundError(
                f"manifest not found: {manifest} — build it with "
                f"`python -m scripts.build_{'busi' if modality == 'us' else 'ddsm'}`"
            )
        self.modality = modality
        self.root = root
        self.image_size = tuple(image_size)
        self.resample = resample or DEFAULT_RESAMPLE[modality]

        df = pd.read_csv(manifest)
        if "risk" not in df.columns:
            raise KeyError(f"{manifest} has no 'risk' column")
        if split:
            if "split" not in df.columns:
                raise KeyError(f"{manifest} has no 'split' column")
            df = df[df["split"] == split]
        df = _drop_masks(df, manifest)
        df = df[df["risk"] != IGNORE_INDEX].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"no usable rows in {manifest} (split='{split}')")
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    @property
    def pos_rate(self) -> float:
        return float(self.df["risk"].mean())

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = os.path.join(self.root, str(row["image_path"]))
        x = to_3ch(load_image(path, self.image_size, self.resample, self.modality))
        y = torch.tensor(int(row["risk"]), dtype=torch.long)
        return x, y


class PretrainDataset(Dataset):
    """The P1 (I-JEPA) pool: images only, no labels, several manifests pooled.

    Two things differ from `LesionDataset`, and both are deliberate:

    * **Nothing is dropped for want of a label.** BUSI's `normal` images carry no
      benign/malignant target and `LesionDataset` discards them — but SSL never
      looks at a label, and a normal breast is exactly the anatomy the encoder
      should learn. Dropping them would throw away 133 free images.
    * **The test split is excluded, always.** This is split rule §2.2: a test
      patient must not appear in *any* phase, pretraining included. The encoder
      seeing a test image unlabeled is still the encoder fitting that image.
    """

    def __init__(
        self,
        manifests,
        modality: str,
        image_size: Tuple[int, int] = (224, 224),
        root: str = "",
        resample: Optional[str] = None,
        exclude_splits: Tuple[str, ...] = ("test",),
        text_encoder=None,
    ):
        if isinstance(manifests, str):
            manifests = [manifests]
        self.modality = modality
        self.root = root
        self.image_size = tuple(image_size)
        self.resample = resample or DEFAULT_RESAMPLE[modality]
        self.text_encoder = text_encoder
        self.cond = None

        frames, conds = [], []
        for m in manifests:
            if not os.path.exists(m):
                raise FileNotFoundError(f"manifest not found: {m}")
            df = pd.read_csv(m)
            if "image_path" not in df.columns:
                raise KeyError(f"{m} has no 'image_path' column")
            n_all = len(df)
            if "split" in df.columns:
                df = df[~df["split"].isin(exclude_splits)]
            else:
                raise KeyError(
                    f"{m} has no 'split' column — refusing to pretrain on it, since "
                    f"test images cannot be held out (§2.2)"
                )
            df = _drop_masks(df, m)
            # Encode BEFORE narrowing to image_path: the text columns are still here.
            # A manifest without those columns transforms to the all-absent vector, which
            # is correct and is exactly the case the warning below is about.
            if text_encoder is not None:
                # The manifest name is passed, not inferred: in precomputed mode the
                # encoder must know which file these rows came from, or it would index a
                # cache built for a different manifest.
                conds.append(text_encoder.transform(df, manifest=m))
            df = df[["image_path"]].copy()
            df["source"] = os.path.basename(m)
            print(f"  {os.path.basename(m):24s} {len(df):6d} / {n_all} images (test excluded)")
            frames.append(df)

        self.df = pd.concat(frames, ignore_index=True)
        if len(self.df) == 0:
            raise ValueError(f"no usable rows in {manifests}")

        if text_encoder is not None:
            import numpy as np
            self.cond = np.concatenate(conds, axis=0).astype("float32")
            assert len(self.cond) == len(self.df)
            self._warn_if_text_marks_the_source()

    def _warn_if_text_marks_the_source(self) -> None:
        """🔴 The failure mode that makes a conditioned pool worse than useless.

        In a mixed pool only some manifests carry text — in us-v2, BUS-CoT does and
        BUSI/BUS-BRA/BUS-UCLM do not. Every row from the others therefore gets the
        `is_absent` vector, and `is_absent` then identifies the source dataset perfectly.
        The predictor can read "this image is from BUS-CoT" straight off the conditioning
        and use it as a free scanner/centre cue, which is not the semantic signal the
        experiment is about — and it will still look like conditioning helped.

        There is no clean fix inside the loader, so this warns rather than pretends.
        The honest options are: pretrain the conditioned arm on the text-carrying manifest
        alone, or drop the text for a random share of BUS-CoT rows so `is_absent` stops
        being a source label.
        """
        import numpy as np
        absent = self.cond[:, -1] > 0.5
        by_src = {}
        for src in self.df["source"].unique():
            m = (self.df["source"] == src).to_numpy()
            by_src[src] = float(absent[m].mean())
        if len(by_src) > 1 and (max(by_src.values()) - min(by_src.values())) > 0.5:
            print("  🔴 conditioning WARNING: is_absent almost perfectly identifies the "
                  "source manifest —")
            for src, r in sorted(by_src.items(), key=lambda kv: -kv[1]):
                print(f"       {src:24s} {100 * r:5.1f}% absent")
            print("       the predictor can use this as a dataset-identity cue rather "
                  "than a semantic one. Pretrain the conditioned arm on the "
                  "text-carrying manifest alone, or randomly drop text so absence "
                  "stops being a source label.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        path = os.path.join(self.root, str(self.df.iloc[idx]["image_path"]))
        img = to_3ch(load_image(path, self.image_size, self.resample, self.modality))
        if self.cond is None:
            return img                       # unchanged: existing runs see exactly this
        return img, torch.from_numpy(self.cond[idx])
