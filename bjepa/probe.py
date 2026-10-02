"""Did P1 actually learn anything? Freeze the encoder, fit a linear head, read AUROC.

    python -m bjepa.probe checkpoints/p1_us_ijepa_armC.pt --manifest manifests/busi.csv
    python -m bjepa.probe --encoder vit_base_patch16_224.augreg_in21k_ft_in1k \
                          --manifest manifests/busi.csv           # the init, un-pretrained

This is P1's only honest scoreboard, and the reason it exists is that the training
loop cannot tell you what you want to know:

* **Loss falling means nothing on its own.** A collapsed encoder — one vector for
  every patch — drives this loss toward zero beautifully. `patch_std` catches the
  total collapse, but not the partial one where representations get blander without
  flatlining.
* **The question P1 is actually asked is downstream.** "Are these features better
  than what I started from?" is not a question about the pretext task. So we ask it
  directly: freeze the encoder, mean-pool the patch tokens, fit one linear layer on
  the train split, score the val split.

The comparison that matters is therefore always a **pair**: the same encoder before
and after I-JEPA. A probe number on its own is unreadable. If continued pretraining
did not move it, P1 did nothing, and no amount of loss-curve staring will change that.

**This is also how the init is chosen** (ImageNet ViT vs USF-MAE vs RAD-DINO, arms C
and D). Not by auditing which foundation model saw which dataset — by probing.

The head is fit on features, so it is a closed-form-ish convex problem; logistic
regression on cached embeddings is enough and takes seconds, no GPU.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .datasets import LesionDataset
from .models.vision_transformer import ViTEncoder, load_mae_checkpoint


def describe_run(ck: dict) -> None:
    """Say what this checkpoint actually is, before anyone reads an AUROC off it.

    `cfg['epochs']` is not evidence of anything — `train_ssl.run()` takes the real
    epoch count as an argument, so a two-minute smoke test carries `epochs: 100` too.
    Checkpoints written since then record a `run` block with the resolved numbers;
    older ones cannot be interrogated at all, and saying so is the honest output.
    """
    run = ck.get("run")
    if not run:
        print("  run: UNKNOWN — this checkpoint predates provenance recording. "
              "cfg['epochs'] is not evidence; only the file timestamp is.")
        return
    line = (f"  run: {run['epochs_done']}/{run['epochs_config']} epochs, "
            f"{run['steps_done']} steps, pool {run['pool']}, {run['device']}")
    if run.get("smoke"):
        line += "   <-- SMOKE RUN, not a result"
    elif not run.get("finished", True):
        line += "   <-- UNFINISHED (periodic save)"
    print(line)


@torch.no_grad()
def embed(encoder: ViTEncoder, ds, dev, batch_size: int = 32) -> tuple:
    """Mean-pool the patch tokens of every image. I-JEPA has no CLS token to use."""
    encoder.eval()
    xs, ys = [], []
    for x, y in DataLoader(ds, batch_size=batch_size, shuffle=False):
        feats = encoder(x.to(dev)).mean(dim=1)        # [B, N, D] -> [B, D]
        xs.append(feats.float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(xs), np.concatenate(ys)


def probe(
    manifest: str,
    modality: str,
    encoder_name: str,
    checkpoint: str = "",
    init_checkpoint: str = "",
    image_size=(224, 224),
    train_split: str = "train",
    val_split: str = "validation",
    train_fracs=(1.0,),
    frac_seeds=(0,),
) -> dict:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = ViTEncoder(name=encoder_name, pretrained=True, img_size=image_size)
    tag = f"{encoder_name} (init only)"
    if init_checkpoint:                     # e.g. USF-MAE, before any I-JEPA
        load_mae_checkpoint(enc, init_checkpoint)
        tag = f"{init_checkpoint} (init only)"
    ck = None
    if checkpoint:                          # after P1 — overwrites whatever init above
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        enc.load_state_dict(ck["encoder"])
        tag = f"{checkpoint} (after P1)"
    enc = enc.to(dev)
    print(f"  encoder: {tag}")
    if ck is not None:
        describe_run(ck)

    tr = LesionDataset(manifest, modality, split=train_split, image_size=image_size)
    va = LesionDataset(manifest, modality, split=val_split, image_size=image_size)
    print(f"  train {len(tr)} / val {len(va)}  (pos rate {tr.pos_rate:.2f} / {va.pos_rate:.2f})")

    xtr, ytr = embed(enc, tr, dev)
    xva, yva = embed(enc, va, dev)

    # Label-efficiency curves: embed once, fit one head per (fraction, subset seed). The
    # encoder is frozen, so the features do not depend on which labels the head sees.
    if tuple(train_fracs) != (1.0,):
        from .datasets import label_subset
        full_df = tr.df.reset_index(drop=True)
        outs = []
        for frac in train_fracs:
            for s_ in (frac_seeds if frac < 1.0 else (0,)):
                sub = label_subset(full_df.assign(_i=range(len(full_df))), frac, s_)
                idx = sub["_i"].to_numpy()
                print(f"  -- train_frac {frac:g}  seed {s_}  n_train {len(idx)} "
                      f"(pos {ytr[idx].mean():.2f})")
                outs.append(_fit_and_record(xtr[idx], ytr[idx], xva, yva, checkpoint, manifest,
                                            train_split, val_split, init_checkpoint, len(idx),
                                            len(va), extra={"train_frac": float(frac),
                                                            "frac_seed": int(s_)}))
        return {"curve": outs}

    return _fit_and_record(xtr, ytr, xva, yva, checkpoint, manifest, train_split, val_split,
                           init_checkpoint, len(tr), len(va))


def _fit_and_record(xtr, ytr, xva, yva, checkpoint, manifest, train_split, val_split,
                    init_checkpoint, n_train, n_val, extra=None) -> dict:
    # Standardise before fitting, or the probe measures the wrong thing. Encoders come
    # out at wildly different feature scales — measured, ImageNet ViT-B sits at
    # feat_std 1.26 and USF-MAE at 0.025, a 50x gap — and a logistic head with a fixed
    # L2 penalty (C) is *not* scale-invariant: on the small-scale features the penalty
    # dominates the likelihood and the head underfits. Skip this and you will conclude
    # that the ultrasound foundation model is worse than ImageNet, when all you have
    # measured is that its activations are smaller.
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=1.0),
    ).fit(xtr, ytr)
    p = clf.predict_proba(xva)[:, 1]
    out = {
        "auroc": float(roc_auc_score(yva, p)),
        "bal_acc": float(balanced_accuracy_score(yva, p > 0.5)),
        "feat_std": float(xva.std(0).mean()),          # blandness check, see docstring
    }
    print(f"  AUROC {out['auroc']:.3f}   bal-acc {out['bal_acc']:.3f}   "
          f"feat_std {out['feat_std']:.3f}")

    # Persist it. Printing alone means the number reaches a document by hand, and by hand
    # is how 0.778 becomes 0.788 and how a smoke run becomes a result. See bjepa/results.py;
    # the table is generated from these with `python -m scripts.make_results`.
    from .results import record
    record("probe", metrics=out, checkpoint=checkpoint, manifest=manifest,
           extra={"train_split": train_split, "val_split": val_split,
                  "init_checkpoint": init_checkpoint or None,
                  "n_train": n_train, "n_val": n_val, **(extra or {})})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default="", help="P1 .pt; omit to probe the raw init")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--modality", default="")
    ap.add_argument("--encoder", default="", help="only needed when probing a raw init")
    ap.add_argument("--init-checkpoint", default="", help="raw FM weights, e.g. USF-MAE .pth")
    ap.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"),
                    help="probe resolution for a raw init. Must match the run's image_size, "
                         "or before/after are measured at different resolutions (MG is 280x210, "
                         "not the 224 default) — a silent confound in the before/after gap.")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="validation")
    ap.add_argument("--train-fracs", type=float, nargs="+", default=[1.0],
                    help="label-efficiency curve: fit one head per fraction of the train labels")
    ap.add_argument("--frac-seeds", type=int, nargs="+", default=[0],
                    help="subset seeds per fraction (bjepa.datasets.label_subset)")
    a = ap.parse_args()

    enc_name, size, modality = a.encoder, (224, 224), a.modality
    if a.checkpoint:
        cfg = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["cfg"]
        enc_name = cfg["model"]["encoder"]
        size = tuple(cfg.get("image_size", [224, 224]))
        modality = modality or cfg["data"]["modality"]
    if a.image_size:                     # explicit override, for raw-init probes at MG's resolution
        size = tuple(a.image_size)
    if not enc_name:
        ap.error("give a checkpoint, or --encoder to probe an un-pretrained init")
    if not modality:
        ap.error("--modality is required when probing a raw init")

    probe(
        manifest=a.manifest,
        modality=modality,
        encoder_name=enc_name,
        checkpoint=a.checkpoint,
        init_checkpoint=a.init_checkpoint,
        image_size=size,
        train_split=a.train_split,
        val_split=a.val_split,
        train_fracs=tuple(a.train_fracs),
        frac_seeds=tuple(a.frac_seeds),
    )


if __name__ == "__main__":
    main()
