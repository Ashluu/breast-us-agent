"""Download BUS-BRA (breast ultrasound) and build manifests/bus_bra.csv.

Source: Zenodo record 8231412 (auth-free, 128 MB, CC BY 4.0).
Gomez-Flores et al., Med Phys 2024. One of BINDS's five public US sets.

Why this set matters more than its size suggests. BUS-BRA is the only open breast-US
set that carries all three things BUSI lacks:

  * `Case` = a real patient ID (1,064 patients over 1,875 images). BUSI ships none, so
    its split can put one patient on both sides and its probe AUROC is inflated —
    plan §2, and the reason no BUSI number may be quoted as a result. BUS-BRA has no
    such debt. Note the -l/-r suffix: those are two views of the SAME case, so an
    image-level split here would leak badly. The official folds already handle it.
  * BI-RADS 2/3/4/5 and biopsy-proven pathology -> P3's BI-RADS head, not just P1.
  * Four scanners (GE Logiq 5/7, Toshiba Aplio 300, U-Systems) -> some device
    diversity, where BUSI is single-centre.

Splits come from the shipped 5-fold CV, not from make_splits: the folds are already
patient-grouped (verified: 0 of 1,064 cases span two folds) and stratified on
pathology (~67/33 in every fold). fold 1 -> test, fold 2 -> validation, 3-5 -> train.
Using the published grouping keeps our numbers comparable to the paper's.

That gives a 385-image validation split against BUSI's 104 — which is the point. At
n=104 the readable difference is ~0.055 AUROC; at n=385 it is ~0.026. BUS-BRA is what
makes a US probe number quotable at all.

**Masks are not extracted.** The archive ships Images/ and Masks/ side by side, 1,875
each. Training on a lesion mask leaks its label through blob geometry — this is exactly
the CBIS-DDSM failure that voided every MG number before `clean_ddsm` (plan §1). Here
we simply never write them to disk.

    python -m scripts.build_bus_bra
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile

import pandas as pd
from PIL import Image

URL = "https://zenodo.org/records/8231412/files/BUSBRA.zip?download=1"
ARCHIVE = os.path.join("data", "bus_bra", "BUSBRA.zip")

RAW_DIR = os.path.join("data", "bus_bra")
IMG_DIR = os.path.join(RAW_DIR, "images")
MANIFEST = os.path.join("manifests", "bus_bra.csv")

# fold -> split. The archive's own 5-fold CV, kept as published.
FOLD_SPLIT = {1: "test", 2: "validation", 3: "train", 4: "train", 5: "train"}
RISK = {"benign": 0, "malignant": 1}


def _fetch() -> str:
    if not os.path.exists(ARCHIVE):
        os.makedirs(RAW_DIR, exist_ok=True)
        print(f"downloading BUS-BRA (~128 MB) -> {ARCHIVE} ...")
        urllib.request.urlretrieve(URL, ARCHIVE)
    return ARCHIVE


def main() -> None:
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)

    z = zipfile.ZipFile(_fetch())
    meta = pd.read_csv(io.BytesIO(z.read("BUSBRA/bus_data.csv")))
    folds = pd.read_csv(io.BytesIO(z.read("BUSBRA/5-fold-cv.csv")))
    df = meta.merge(folds[["ID", "kFold"]], on="ID", validate="one_to_one")

    # The claim this file rests on. If a future revision of the archive breaks it, the
    # patient-level split silently becomes an image-level one — fail loudly instead.
    spanning = (df.groupby("Case")["kFold"].nunique() > 1).sum()
    if spanning:
        raise ValueError(
            f"{spanning} cases span more than one fold — the shipped folds are not "
            f"patient-grouped and cannot be used as splits (plan §2.1)"
        )

    rows = []
    for _, r in df.iterrows():
        rel = f"data/bus_bra/images/{r['ID']}.png"
        # 'L' to match build_busi.py; the archive is already single-channel.
        Image.open(io.BytesIO(z.read(f"BUSBRA/Images/{r['ID']}.png"))).convert("L").save(rel)
        rows.append({
            "image_path": rel,
            "label": RISK[r["Pathology"]],
            "split": FOLD_SPLIT[int(r["kFold"])],
            "birads": int(r["BIRADS"]),
            "risk": RISK[r["Pathology"]],
            "patient_id": int(r["Case"]),
        })

    man = pd.DataFrame(rows)
    man.to_csv(MANIFEST, index=False)

    print(f"wrote {len(man)} images -> {IMG_DIR}  (masks deliberately not extracted)")
    print(f"wrote manifest -> {MANIFEST}")
    print("split:", man["split"].value_counts().to_dict())
    print("pathology:", man["risk"].value_counts().to_dict())
    print("BI-RADS:", man["birads"].value_counts().sort_index().to_dict())
    print(f"patients: {man['patient_id'].nunique()} (patient-grouped folds, as published)")
    va = man[man["split"] == "validation"]
    print(f"validation: {len(va)} images / {va['patient_id'].nunique()} patients "
          f"— BUSI's is 104 images with no patient IDs at all")


if __name__ == "__main__":
    main()
