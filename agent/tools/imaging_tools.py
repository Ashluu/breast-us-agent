"""LangChain tool wrappers around the frozen P1 encoders -- the only new "model" code in
this phase. Everything here is read-only inference: no training, no encoder is ever put in
.train() mode, and -- per P3_AGENT_NOTES.md -- no tool ever hands the LLM freeform
diagnostic prose to relay, only numbers/categories it can reason over, plus one tool
(generate_report) that turns those into a fixed-template report.

Loading an encoder and fitting its linear probe both cost real time (fitting means a full
forward pass over the train split), so both are cached at module scope -- fit once per
process, reused across every call. A ReAct loop that calls call_us five times in one case
must not re-fit the probe five times.
"""

from __future__ import annotations

import io
import json
import os
import zipfile

import numpy as np
import torch
from langchain_core.tools import tool
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bjepa.datasets import LesionDataset
from bjepa.models.vision_transformer import ViTEncoder
from bjepa.probe import embed
from bjepa.transforms import load_image, to_3ch

from ..config import CONFIG

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Module-level caches, keyed by modality ("us" / "mg"). Populated lazily on first use so
# importing this module (e.g. just to build the graph) doesn't eagerly load anything.
_ENCODERS: dict = {}
_PROBES: dict = {}
_SIZES: dict = {}   # (H, W) each encoder was trained at, read from its checkpoint cfg


def _checkpoint_and_manifest(modality: str) -> tuple[str, str]:
    if modality == "us":
        return CONFIG.us_checkpoint, CONFIG.us_manifest
    if modality == "mg":
        return CONFIG.mg_checkpoint, CONFIG.mg_manifest
    raise ValueError(f"unknown modality '{modality}', expected 'us' or 'mg'")


def _get_encoder(modality: str) -> ViTEncoder:
    if modality in _ENCODERS:
        return _ENCODERS[modality]
    checkpoint, _ = _checkpoint_and_manifest(modality)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"{checkpoint} not found on this machine. This tool needs the actual trained "
            f"P1 checkpoint (see P3_AGENT_NOTES.md 'Open TODO') -- copy it here or point "
            f"agent/config.py at wherever it actually lives (this Mac, natively, or UHEM)."
        )
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # Resolve name + input size from the checkpoint's own cfg, exactly as bjepa/probe.py
    # and probe_longtail.py do. The MG arm-C checkpoint is RAD-DINO at 280x210 patch-14, not
    # the timm ViT-B/16 at 224x224 that config.py defaults to; building the wrong one fails
    # loudly on load_state_dict here, but a same-shape mismatch would silently void a result.
    cfg = ck.get("cfg", {}) or {}
    name = cfg.get("model", {}).get("encoder") or CONFIG.encoder_name
    size = tuple(cfg.get("image_size") or CONFIG.image_size)
    enc = ViTEncoder(name=name, pretrained=True, img_size=size)
    enc.load_state_dict(ck["encoder"])
    enc = enc.to(_DEV).eval()
    _ENCODERS[modality] = enc
    _SIZES[modality] = size
    return enc


def _get_probe(modality: str):
    """Fit once per process, same recipe as bjepa/probe.py: mean-pooled patch tokens ->
    StandardScaler -> LogisticRegression, fit on the SAME train split the thesis's own
    AUROC numbers use (RESULT_*.md), so a probe result here is directly comparable, not a
    second, differently-fit number that happens to share a name."""
    if modality in _PROBES:
        return _PROBES[modality]
    enc = _get_encoder(modality)
    _, manifest = _checkpoint_and_manifest(modality)
    tr = LesionDataset(manifest, modality, split=CONFIG.train_split, image_size=_SIZES[modality])
    xtr, ytr = embed(enc, tr, _DEV)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    clf.fit(xtr, ytr)
    _PROBES[modality] = clf
    return clf


def _resolve(image_path: str) -> str:
    """The orchestrator LLM sometimes rewrites a repo-relative path as absolute
    ("data/x.png" -> "/data/x.png"). Undo exactly that, and only when the literal path does
    not exist and the stripped one does; anything else is left alone to fail loudly."""
    if not os.path.exists(image_path) and image_path.startswith("/") \
            and os.path.exists(image_path.lstrip("/")):
        return image_path.lstrip("/")
    return image_path


def _predict_one(image_path: str, modality: str) -> dict:
    image_path = _resolve(image_path)
    enc = _get_encoder(modality)
    clf = _get_probe(modality)
    resample = "bilinear" if modality == "us" else "nearest"  # DEFAULT_RESAMPLE, bjepa/transforms.py
    x = to_3ch(load_image(image_path, _SIZES[modality], resample, modality))
    x = x.unsqueeze(0).to(_DEV)
    with torch.no_grad():
        feat = enc(x).mean(dim=1).float().cpu().numpy()   # [1, D] -- same pooling as bjepa/probe.embed
    prob = float(clf.predict_proba(feat)[0, 1])
    return {
        "modality": modality,
        "prob_malignant": round(prob, 4),
        "confidence": round(abs(prob - 0.5) * 2, 4),   # 0 = coin flip, 1 = certain either way
    }


@tool
def call_us(image_path: str) -> str:
    """Run the frozen ultrasound I-JEPA encoder + linear probe on one ultrasound image.
    Input: path to a BUS-BRA image (see manifests/bus_bra.csv). Returns a JSON object with
    malignancy probability and a confidence score -- NOT the raw embedding: that's 768
    floats, useless in a chat context, and it never leaves this process."""
    return json.dumps(_predict_one(image_path, "us"))


@tool
def call_mg(image_path: str) -> str:
    """Run the frozen mammography I-JEPA encoder + linear probe on one mammography image.
    Same contract as call_us, mammography side."""
    return json.dumps(_predict_one(image_path, "mg"))


def _busbra_mask_bytes(image_path: str):
    """BUS-BRA ships its own lesion masks inside the same zip the images came from --
    identical lookup to scripts/build_bus_bra_roi.py. Returns None if this image has no
    mask; the caller decides what "no mask" means, this function never guesses one."""
    archive = os.path.join("data", "bus_bra", "BUSBRA.zip")
    if not os.path.exists(archive):
        return None
    base = os.path.basename(image_path)
    mask_name = "BUSBRA/Masks/" + base.replace("bus_", "mask_", 1)
    with zipfile.ZipFile(archive) as z:
        if mask_name not in z.namelist():
            return None
        return z.read(mask_name)


@tool
def segment_lesion(image_path: str) -> str:
    """Find the lesion bounding box for an ultrasound image, using BUS-BRA's own shipped
    mask. The mask is used ONLY to compute a bbox here -- never returned to the LLM or fed
    to a model, same safety argument as crop_to_lesion in bjepa/transforms.py. Mammography
    has no mask source wired up yet -- returns bbox: null rather than fabricating one."""
    from PIL import Image

    mask_bytes = _busbra_mask_bytes(image_path)
    if mask_bytes is None:
        return json.dumps({
            "bbox": None, "mask_area_fraction": None,
            "note": "no mask found for this image (wrong modality, or not in BUSBRA.zip)",
        })
    image_path = _resolve(image_path)
    img = Image.open(image_path).convert("L")
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.NEAREST)
    arr = np.asarray(mask)
    ys, xs = np.where(arr > 0)
    if len(ys) == 0:
        return json.dumps({"bbox": None, "mask_area_fraction": 0.0, "note": "mask present but empty"})
    w, h = img.size
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return json.dumps({"bbox": bbox, "mask_area_fraction": round(len(ys) / (w * h), 4)})


@tool
def generate_report(findings_json: str | dict) -> str:
    """Compose the final report from structured findings ONLY -- this fills a fixed
    template, it never lets the LLM write free diagnostic prose. Same reasoning as why
    BUS-CoT's `reasoning_response` field was rejected as a text-JEPA target
    (see DATA_BUS_COT.md: it leaked the histopathology label straight into generated
    sentences) -- letting an LLM freely narrate a "report" here would reopen that leak one
    layer up the stack.

    Input: a JSON object (or a JSON string of one) with any subset of: modality,
    prob_malignant, confidence, bbox, mask_area_fraction, birads, and (from agent/loop.py)
    n_views, view_spread, stable, uses_expert_mask. Fields not present are omitted, never
    guessed."""
    # Typed `str | dict` on purpose: LFM2.5 emits the findings as a JSON object, and a plain
    # `str` schema made every call fail validation, so the orchestrator retried until the
    # recursion limit instead of ever finishing a case.
    if isinstance(findings_json, dict):
        f = findings_json
    else:
        try:
            f = json.loads(findings_json)
        except json.JSONDecodeError:
            return ('ERROR: generate_report needs a JSON object of findings, e.g. '
                    '{"modality": "us", "prob_malignant": 0.82}')

    # Sentence templates, not key:value lines -- reads like an actual report paragraph for
    # the paper's case studies, while still being 100% deterministic string formatting. The
    # LLM supplies NUMBERS (via the findings dict); it never supplies WORDS beyond selecting
    # which of these fixed sentences apply. That boundary is the whole safety property this
    # function exists for -- widen the vocabulary here if you like, but never let a value
    # come from anywhere except the findings dict.
    sentences = []

    modality_name = {"us": "ultrasound", "mg": "mammography"}.get(f.get("modality"), f.get("modality"))
    if modality_name and "prob_malignant" in f:
        p = f["prob_malignant"]
        leaning = "malignant-leaning" if p >= 0.5 else "benign-leaning"
        sentences.append(
            f"This {modality_name} case was assessed by the frozen encoder and linear probe, "
            f"which estimate a {p:.0%} probability of malignancy ({leaning})."
        )
    elif modality_name:
        sentences.append(f"This {modality_name} case was assessed, but no malignancy probability was recorded.")

    if "confidence" in f:
        c_val = f["confidence"]
        certainty = "high" if c_val >= 0.7 else "moderate" if c_val >= 0.3 else "low"
        sentences.append(f"Model confidence in this estimate is {certainty} ({c_val:.2f} on a 0-1 scale).")

    if f.get("bbox") and f.get("mask_area_fraction") is not None:
        sentences.append(
            f"The lesion occupies about {f['mask_area_fraction']:.0%} of the frame, "
            f"within the bounding box {f['bbox']} (pixels)."
        )
    elif f.get("bbox"):
        sentences.append(f"The lesion bounding box is {f['bbox']} (pixels).")

    # Written by agent/loop.py: the estimate above is then the mean over several views.
    if f.get("n_views") and "view_spread" in f:
        verdict = "the views agree" if f.get("stable") else "the views disagree, so treat the estimate as unstable"
        sentences.append(
            f"The estimate combines {f['n_views']} views of the image, which differ by at most "
            f"{f['view_spread']:.0%} in probability ({verdict})."
        )
    if f.get("uses_expert_mask"):
        sentences.append(
            "Lesion-centred views were placed with the dataset's expert mask, which a deployed "
            "system would not have."
        )

    if "birads" in f:
        sentences.append(f"The recorded BI-RADS category is {f['birads']}.")

    if not sentences:
        sentences.append("No findings were provided to summarize.")

    sentences.append(
        "This is a pilot-system output for a course/thesis exercise, not a clinical "
        "decision -- see P3_AGENT_NOTES.md for stated limitations."
    )
    return " ".join(sentences)


TOOLS = [call_us, call_mg, segment_lesion, generate_report]
