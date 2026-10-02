"""Propose -> score -> feed back -> refine: a closed loop for the P3 agent, after CLARITY
(Ding et al., arXiv:2512.08029).

CLARITY has an LLM policy propose candidate therapies, a frozen latent world model score
each one (survival risk), and the scores go back into the next prompt for K rounds. This
module keeps that shape and changes what the pieces are, because this project has one
image per lesion and no treatment or follow-up data:

    CLARITY                         here
    -----------------------------   ------------------------------------------------
    action = a therapy              action = a VIEW of the same image (full frame,
                                    lesion crops, box geometry), see VIEWS
    Actor = latent world model      scorer = frozen I-JEPA encoder + one linear probe
                                    per view, fit on BUS-BRA train with that view
    feedback = survival risk        feedback = each view's malignancy probability and
                                    how far the views disagree
    objective = lowest risk         objective = a STABLE estimate

The objective is the one deliberate departure. Picking the view with the lowest (or most
confident) malignancy probability would be choosing the answer, not measuring it. So the
loop never selects a view: every scored view enters the final estimate (mean of logits),
and what the rounds buy is evidence on whether that estimate holds across views. It stops
when the views agree (spread below `stable_spread`), after `k` rounds, or when the menu is
empty.

The LLM only chooses view NAMES from a fixed menu. Its proposal is checked in code
(`_constrain`, CLARITY's constraint set) and replaced by a fixed order when it is invalid;
which of the two happened is recorded per round. The LLM sees probabilities, never the
image and never the label, and its one-line reason goes to the trace only. The report
still comes from generate_report's template.

Lesion-centred views and the box view need a lesion mask to place the box. On BUS-BRA
that is the expert mask, which a deployed system would not have; the result says so
(`uses_expert_mask`). Without a mask only `full` and `full_flip` are on the menu.

    python -m agent.loop --n 5                 # LLM policy on 5 validation cases -> agent/traces/
    python -m agent.loop --eval                # validation table: single views vs the loop
    python -m agent.loop --eval --llm-n 40     # same, plus the LLM policy on 40 cases
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
import zipfile
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from bjepa.models.vision_transformer import ViTEncoder
from bjepa.probe import embed
from bjepa.transforms import RESAMPLE, to_3ch
from scripts.build_bus_bra_roi import crop_to_lesion
from scripts.roi_box_baseline import ARCHIVE, FEATURES, box_features_from_mask

from .config import CONFIG

BOX_ONLY = [f for f in FEATURES if f != "mask_fill"]   # mask_fill needs the outline, not just the box
CACHE_DIR = os.path.join(".cache", "agent_loop")


@dataclass(frozen=True)
class View:
    probe: str            # which fitted probe scores this view
    needs_mask: bool
    describe: str         # the menu line the LLM reads
    margin: float = 0.15
    square: bool = False
    flip: bool = False
    pixels: bool = True


VIEWS = {
    "full": View("full", False, "the whole ultrasound frame"),
    "full_flip": View("full", False, "the whole frame mirrored left-right", flip=True),
    "roi": View("roi", True, "crop to the lesion box with a small margin"),
    "roi_wide": View("roi_wide", True, "crop to the lesion box with wide surrounding tissue", margin=0.5),
    "roi_sq": View("roi_sq", True, "square crop around the lesion, no stretching", square=True),
    "box": View("box", True, "lesion box size, shape and position only, no pixels", pixels=False),
}
# Used when the LLM's proposal is invalid, and as the fixed policy in --eval.
FALLBACK_ORDER = ["full", "roi", "box", "roi_sq", "roi_wide", "full_flip"]


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def _sigmoid(z: float) -> float:
    return float(1 / (1 + np.exp(-z)))


class _Masks:
    """BUS-BRA's own masks, read from the archive. Used only to place a box."""

    def __init__(self):
        self._zip = zipfile.ZipFile(ARCHIVE) if os.path.exists(ARCHIVE) else None
        self._names = set(self._zip.namelist()) if self._zip else set()

    def get(self, image_path: str) -> Optional[Image.Image]:
        name = "BUSBRA/Masks/" + os.path.basename(image_path).replace("bus_", "mask_", 1)
        if name not in self._names:
            return None
        mask = Image.open(io.BytesIO(self._zip.read(name))).convert("L")
        return mask if np.asarray(mask).any() else None


class _ViewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, view: View, scorer: "ViewScorer"):
        self.df, self.view, self.scorer = df.reset_index(drop=True), view, scorer

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # A train image without a mask stays uncropped, as in scripts/build_bus_bra_roi.py.
        x = self.scorer.view_tensor(row["image_path"], self.view, allow_uncropped=True)
        return x, torch.tensor(int(row["risk"]), dtype=torch.long)


class ViewScorer:
    """Frozen encoder plus one probe per view. The probe recipe is bjepa/probe.py's
    (mean-pooled patch tokens -> StandardScaler -> LogisticRegression, C=1, fit on train);
    the only thing that changes between probes is the view the train images were given in."""

    def __init__(self, checkpoint: str = CONFIG.us_checkpoint, manifest: str = CONFIG.us_manifest):
        self.checkpoint, self.manifest = checkpoint, manifest
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = ck.get("cfg", {}) or {}
        name = cfg.get("model", {}).get("encoder") or CONFIG.encoder_name
        self.size = tuple(cfg.get("image_size") or CONFIG.image_size)
        enc = ViTEncoder(name=name, pretrained=True, img_size=self.size)
        enc.load_state_dict(ck["encoder"])
        self.enc = enc.to(self.dev).eval()
        self.masks = _Masks()
        self.df = pd.read_csv(manifest)
        self._probes: dict = {}
        st = os.stat(checkpoint)
        self._tag = f"{os.path.splitext(os.path.basename(checkpoint))[0]}_{st.st_size}_{int(st.st_mtime)}"

    # ---- one image, one view -------------------------------------------------------
    def view_tensor(self, image_path: str, view: View, allow_uncropped: bool = False):
        img = Image.open(image_path).convert("L")
        if view.needs_mask:
            mask = self.masks.get(image_path)
            if mask is None and not allow_uncropped:
                return None
            if mask is not None:
                img = crop_to_lesion(img, mask, view.margin, view.square)
        if view.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.resize(self.size[::-1], RESAMPLE["bilinear"])     # same as load_image for US
        return to_3ch(torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None, ...])

    def box_vector(self, image_path: str):
        mask = self.masks.get(image_path)
        if mask is None:
            return None
        f = box_features_from_mask(mask, Image.open(image_path).size)
        return np.array([[f[k] for k in BOX_ONLY]], dtype=np.float64)

    def available(self, image_path: str) -> list:
        has_mask = self.masks.get(image_path) is not None
        return [n for n, v in VIEWS.items() if has_mask or not v.needs_mask]

    # ---- features and probes -------------------------------------------------------
    def _features(self, view_name: str, split: str) -> tuple:
        """Features of one split under one view, cached on disk per checkpoint."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = os.path.join(CACHE_DIR, f"{self._tag}_{view_name}_{split}.npz")
        if os.path.exists(path):
            z = np.load(path)
            return z["x"], z["y"]
        view, df = VIEWS[view_name], self.df[self.df["split"] == split]
        if view.pixels:
            x, y = embed(self.enc, _ViewDataset(df, view, self), self.dev)
        else:
            rows = [self.box_vector(p) for p in df["image_path"]]
            keep = [r is not None for r in rows]
            x = np.concatenate([r for r in rows if r is not None])
            y = df["risk"].to_numpy()[keep]
            if split != "train" and not all(keep):
                raise ValueError(f"{split}: {keep.count(False)} images have no mask, box rows would not line up")
        np.savez(path, x=x, y=y)
        return x, y

    def probe(self, key: str):
        if key not in self._probes:
            x, y = self._features(key, "train")
            self._probes[key] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0)).fit(x, y)
        return self._probes[key]

    def score(self, image_path: str, view_name: str) -> Optional[float]:
        """Malignancy probability of one image under one view, or None if the view needs a
        mask this image does not have."""
        view = VIEWS[view_name]
        if view.pixels:
            x = self.view_tensor(image_path, view)
            if x is None:
                return None
            with torch.no_grad():
                feat = self.enc(x.unsqueeze(0).to(self.dev)).mean(dim=1).float().cpu().numpy()
        else:
            feat = self.box_vector(image_path)
            if feat is None:
                return None
        return float(self.probe(view.probe).predict_proba(feat)[0, 1])

    def score_split(self, split: str) -> pd.DataFrame:
        """Every view's probability for every image of a split (BUS-BRA: all have masks)."""
        df = self.df[self.df["split"] == split].reset_index(drop=True)
        out = df[["image_path", "risk"]].copy()
        for name, view in VIEWS.items():
            x, _ = self._features(name, split)
            out[name] = self.probe(view.probe).predict_proba(x)[:, 1]
        return out


# ---- policies: (state) -> (proposed names, source, raw text) ---------------------------
POLICY_SYSTEM = """You plan which views of ONE breast ultrasound image a frozen scoring model \
should look at next. You never see the image and you never state a diagnosis. The goal is a \
stable estimate: pick the views that best test whether the current estimate holds. \
Reply with one JSON object and nothing else: {"next": ["<view name>", ...], "reason": "<at most 15 words>"}"""


def _policy_prompt(state: dict) -> str:
    lines = ["Views you can still choose (use the exact names):"]
    lines += [f"- {n}: {VIEWS[n].describe}" for n in state["remaining"]]
    if state["scores"]:
        lines.append("\nFeedback from the scoring model so far (probability of malignancy):")
        lines += [f"- {n}: {p:.2f}" for n, p in state["scores"].items()]
        lines.append(f"Combined estimate {state['aggregate']:.2f}. "
                     f"Largest disagreement between views {state['spread']:.2f} "
                     f"(below {state['stable_spread']:.2f} counts as stable).")
    else:
        lines.append("\nNothing has been scored yet.")
    lines.append(f"\nChoose up to {state['per_round']} views for round {state['round']}.")
    return "\n".join(lines)


def fixed_policy(state: dict) -> tuple:
    return [n for n in FALLBACK_ORDER if n in state["remaining"]][:state["per_round"]], "fixed", ""


def make_llm_policy() -> Callable:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama

    # temperature 0: the same feedback gives the same proposal, so a trace can be re-run.
    llm = ChatOllama(model=CONFIG.ollama_model, base_url=CONFIG.ollama_base_url, temperature=0)

    def policy(state: dict) -> tuple:
        raw = llm.invoke([SystemMessage(content=POLICY_SYSTEM),
                          HumanMessage(content=_policy_prompt(state))]).content
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            nxt = json.loads(m.group(0)).get("next", []) if m else []
        except (ValueError, AttributeError):
            nxt = []
        return [str(n) for n in nxt] if isinstance(nxt, list) else [], "llm", raw

    return policy


def _constrain(proposed: list, remaining: list, per_round: int) -> tuple:
    """The constraint set: a known view, available for this image, not scored before."""
    accepted, rejected = [], []
    for n in proposed:
        if n not in VIEWS:
            rejected.append([n, "unknown view"])
        elif n not in remaining:
            rejected.append([n, "already scored or needs a mask this image lacks"])
        elif n in accepted:
            rejected.append([n, "repeated"])
        elif len(accepted) >= per_round:
            rejected.append([n, "over the per-round limit"])
        else:
            accepted.append(n)
    return accepted, rejected


def run_loop(score_fn: Callable, available: list, policy: Callable = fixed_policy,
             k: int = 3, per_round: int = 2, stable_spread: float = 0.10) -> dict:
    """One case. `score_fn(view_name) -> probability`; `available` is the menu for this image."""
    scores, rounds, stop = {}, [], "max_rounds"
    for r in range(1, k + 1):
        remaining = [n for n in available if n not in scores]
        if not remaining:
            stop = "menu_exhausted"
            break
        state = {"round": r, "remaining": remaining, "scores": dict(scores), "per_round": per_round,
                 "stable_spread": stable_spread,
                 "aggregate": _sigmoid(np.mean([_logit(p) for p in scores.values()])) if scores else None,
                 "spread": max(scores.values()) - min(scores.values()) if scores else None}
        proposed, source, raw = policy(state)
        accepted, rejected = _constrain(proposed, remaining, per_round)
        if not accepted:
            accepted, source = fixed_policy(state)[0], "fallback"
        for n in accepted:
            scores[n] = score_fn(n)
        spread = max(scores.values()) - min(scores.values())
        rounds.append({"round": r, "source": source, "proposed": proposed, "accepted": accepted,
                       "rejected": rejected, "llm_text": raw,
                       "scores": {n: round(scores[n], 4) for n in accepted},
                       "aggregate": round(_sigmoid(np.mean([_logit(p) for p in scores.values()])), 4),
                       "spread": round(spread, 4)})
        if len(scores) >= 2 and spread < stable_spread:
            stop = "stable"
            break
    prob = _sigmoid(np.mean([_logit(p) for p in scores.values()]))
    spread = max(scores.values()) - min(scores.values())
    findings = {"modality": "us", "prob_malignant": round(prob, 4),
                "confidence": round(abs(prob - 0.5) * 2, 4),
                "n_views": len(scores), "view_spread": round(spread, 4),
                "stable": bool(len(scores) >= 2 and spread < stable_spread),
                "uses_expert_mask": any(VIEWS[n].needs_mask for n in scores)}
    return {"findings": findings, "views": {n: round(p, 4) for n, p in scores.items()},
            "rounds": rounds, "stop": stop}


def assess(image_path: str, scorer: ViewScorer, policy: Callable = fixed_policy, **kw) -> dict:
    """Run the loop on one image and compose the report from the template."""
    from .tools.imaging_tools import _resolve, generate_report

    image_path = _resolve(image_path)
    out = run_loop(lambda n: scorer.score(image_path, n), scorer.available(image_path), policy, **kw)
    out["image_path"] = image_path
    out["report"] = generate_report.invoke({"findings_json": out["findings"]})
    return out


# ---- CLI ---------------------------------------------------------------------------
def _eval(scorer: ViewScorer, args) -> None:
    """Validation only: is the looped estimate better than one view, and what does it cost?"""
    t = scorer.score_split(CONFIG.val_split)
    y = t["risk"].to_numpy()
    res = {"checkpoint": scorer.checkpoint, "split": CONFIG.val_split, "n": len(t),
           "k": args.k, "per_round": args.per_round, "stable_spread": args.stable_spread,
           "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "single_view": {}, "loop": {}}
    print(f"\n{CONFIG.val_split} n={len(t)}  checkpoint {scorer.checkpoint}\n\nsingle view            AUROC")
    for n in VIEWS:
        res["single_view"][n] = float(roc_auc_score(y, t[n]))
        print(f"  {n:18s}  {res['single_view'][n]:.3f}")

    def looped(rows: pd.DataFrame, policy: Callable, menu: list) -> tuple:
        outs = [run_loop(lambda n, row=row: float(row[n]), menu, policy, args.k, args.per_round,
                         args.stable_spread) for _, row in rows.iterrows()]
        return outs, {
            "auroc": float(roc_auc_score(rows["risk"], [o["findings"]["prob_malignant"] for o in outs])),
            "mean_views": float(np.mean([o["findings"]["n_views"] for o in outs])),
            "stable_share": float(np.mean([o["findings"]["stable"] for o in outs])),
            "n": len(rows)}

    # Reference for the loop: every view averaged, no rounds and no early stop.
    every = [_sigmoid(np.mean([_logit(float(row[n])) for n in VIEWS])) for _, row in t.iterrows()]
    res["all_views_mean"] = float(roc_auc_score(y, every))
    print(f"  {'mean of all ' + str(len(VIEWS)):18s}  {res['all_views_mean']:.3f}")

    print("\nloop                               AUROC   views  stable")
    menus = {"fixed, all views": list(VIEWS), "fixed, no mask (full, full_flip)": ["full", "full_flip"]}
    for name, menu in menus.items():
        _, res["loop"][name] = looped(t, fixed_policy, menu)
        m = res["loop"][name]
        print(f"  {name:32s} {m['auroc']:.3f}   {m['mean_views']:.2f}   {m['stable_share']:.0%}")

    if args.llm_n:
        sub = t.sample(n=min(args.llm_n, len(t)), random_state=args.seed)
        outs, m = looped(sub, make_llm_policy(), list(VIEWS))
        src = [r["source"] for o in outs for r in o["rounds"]]
        m["llm_rounds"], m["fallback_rounds"] = src.count("llm"), src.count("fallback")
        _, m_fixed = looped(sub, fixed_policy, list(VIEWS))
        res["loop"]["llm, all views (subset)"], res["loop"]["fixed, all views (same subset)"] = m, m_fixed
        for name in ("llm, all views (subset)", "fixed, all views (same subset)"):
            mm = res["loop"][name]
            print(f"  {name:32s} {mm['auroc']:.3f}   {mm['mean_views']:.2f}   {mm['stable_share']:.0%}   (n={mm['n']})")
        print(f"  LLM proposals accepted in {m['llm_rounds']} of {len(src)} rounds, "
              f"fixed order used in {m['fallback_rounds']}")

    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", f"{time.strftime('%Y%m%d-%H%M%S')}_agent_loop.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5, help="how many validation cases to run")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--policy", default="llm", choices=["llm", "fixed"])
    ap.add_argument("--checkpoint", default=CONFIG.us_checkpoint)
    ap.add_argument("--k", type=int, default=3, help="maximum rounds")
    ap.add_argument("--per-round", type=int, default=2, help="views scored per round")
    ap.add_argument("--stable-spread", type=float, default=0.10)
    ap.add_argument("--eval", action="store_true", help="validation table instead of single cases")
    ap.add_argument("--llm-n", type=int, default=0, help="with --eval: also run the LLM policy on this many cases")
    args = ap.parse_args()

    scorer = ViewScorer(args.checkpoint)
    if args.eval:
        _eval(scorer, args)
        return

    policy = make_llm_policy() if args.policy == "llm" else fixed_policy
    df = scorer.df[scorer.df["split"] == CONFIG.val_split]
    os.makedirs("agent/traces", exist_ok=True)
    for _, row in df.sample(n=min(args.n, len(df)), random_state=args.seed).iterrows():
        out = assess(row["image_path"], scorer, policy, k=args.k, per_round=args.per_round,
                     stable_spread=args.stable_spread)
        out["true_risk"] = int(row["risk"])       # for scoring by hand; never shown to the LLM
        stem = os.path.splitext(os.path.basename(row["image_path"]))[0]
        with open(os.path.join("agent", "traces", f"loop_{stem}.json"), "w") as fh:
            json.dump(out, fh, indent=2)
        f = out["findings"]
        print(f"{stem}: p={f['prob_malignant']:.2f} views={f['n_views']} spread={f['view_spread']:.2f} "
              f"stop={out['stop']} sources={[r['source'] for r in out['rounds']]}")


if __name__ == "__main__":
    main()
