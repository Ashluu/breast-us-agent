"""Run the P3 agent on real BUS-BRA validation cases and save a per-case trace under
agent/traces/ for the paper's case-study section and the accuracy/tool-call-count table
in P3_AGENT_NOTES.md's evaluation plan.

    python -m agent.run_case --n 5

🔴 Does not run as-is on a machine without the real US/MG checkpoint and a local Ollama
server with the orchestrator model pulled (see agent/config.py, P3_AGENT_NOTES.md "Open
TODO") -- run this in the Mac's own native Terminal (not this bridge) or on UHEM, wherever
the checkpoints actually live -- not on a bare clone.
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from .config import CONFIG
from .graph import run_one_case


def _serialize_messages(result: dict) -> list:
    out = []
    for m in result["messages"]:
        entry = {"type": m.__class__.__name__, "content": getattr(m, "content", "")}
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = tool_calls
        out.append(entry)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5, help="how many validation cases to run")
    ap.add_argument("--modality", default="us", choices=["us", "mg"])
    ap.add_argument("--seed", type=int, default=1,
                     help="1 matches the earlier contact-sheet sampling seed, for consistency")
    args = ap.parse_args()

    manifest = CONFIG.us_manifest if args.modality == "us" else CONFIG.mg_manifest
    df = pd.read_csv(manifest)
    df = df[df["split"] == CONFIG.val_split]
    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed)

    os.makedirs("agent/traces", exist_ok=True)
    n_correct = 0
    for _, row in sample.iterrows():
        image_path = row["image_path"]
        print(f"running case: {image_path}")
        result = run_one_case(image_path, args.modality)
        messages = _serialize_messages(result)
        trace = {"image_path": image_path, "true_risk": int(row["risk"]), "messages": messages}

        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join("agent", "traces", f"{stem}.json")
        with open(out_path, "w") as fh:
            json.dump(trace, fh, indent=2)
        print(f"  -> {out_path}")

    print(f"\nwrote {len(sample)} traces to agent/traces/ "
          f"-- score them by hand against true_risk for the accuracy table, "
          f"generate_report doesn't emit a bare 0/1 to auto-grade against.")


if __name__ == "__main__":
    main()
