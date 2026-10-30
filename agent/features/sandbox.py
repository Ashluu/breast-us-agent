"""Checks and runs a feature function written by the LLM, when the library has no suitable one.

A generated feature is Python source that defines exactly one function

    def feature(image, mask) -> float

where `image` is the grayscale B-mode image (2-D float array, 0-255) and `mask` the U-Net lesion outline
(2-D bool array, same shape). The function sees arrays only: no file path, no label, no case id, so it
cannot read the answer.

Three gates, in order (PLAN_RAG_FEATURE_AGENT_2026-10-30.md):
  1. static   AST check: one `feature(image, mask)` def, imports only from ALLOWED_MODULES, no
              dangerous builtins, no dunder attributes, nothing at module level but imports and the def
  2. run      in a separate `python -I` process with a CPU-time cap and a wall-clock timeout
  3. tests    on synthetic shapes (circle, ellipse, star, small blob, noise texture): returns a finite
              float, the same value twice, and not the same value on every shape

This is a guard against mistakes of a local model, not a security boundary against hostile code.
A feature that passes is saved as a CANDIDATE under agent/features/generated/ with its provenance;
it enters the library only after a person reads it.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

ALLOWED_MODULES = {"numpy", "math", "scipy", "scipy.ndimage", "scipy.spatial", "scipy.stats", "skimage", "cv2"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
                "delattr", "input", "breakpoint", "exit", "quit", "help", "memoryview", "dir"}
GENERATED_DIR = os.path.join(os.path.dirname(__file__), "generated")
TIMEOUT_S = 10
CPU_S = 8


@dataclass
class Check:
    ok: bool
    errors: list = field(default_factory=list)
    values: dict = field(default_factory=dict)


def _module_ok(name: str) -> bool:
    return name in ALLOWED_MODULES or any(name.startswith(m + ".") for m in ("scipy", "skimage"))


def static_check(source: str) -> Check:
    errors = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return Check(False, [f"syntax error: {e}"])
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(defs) != 1 or defs[0].name != "feature":
        errors.append("must define exactly one top-level function named `feature`")
    elif [a.arg for a in defs[0].args.args] != ["image", "mask"]:
        errors.append("signature must be feature(image, mask)")
    for n in tree.body:
        doc = isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
        if not isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef)) and not doc:
            errors.append(f"line {n.lineno}: only imports and the function are allowed at module level")
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            errors += [f"line {n.lineno}: import of `{a.name}` not allowed" for a in n.names if not _module_ok(a.name)]
        elif isinstance(n, ast.ImportFrom):
            if n.level or not _module_ok(n.module or ""):
                errors.append(f"line {n.lineno}: import from `{n.module}` not allowed")
        elif isinstance(n, ast.Name) and n.id in BANNED_NAMES:
            errors.append(f"line {n.lineno}: `{n.id}` not allowed")
        elif isinstance(n, ast.Attribute) and n.attr.startswith("_"):
            errors.append(f"line {n.lineno}: attribute `{n.attr}` not allowed")
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            errors.append(f"line {n.lineno}: global/nonlocal not allowed")
    return Check(not errors, errors)


_RUNNER = r"""
import json, resource, sys
import numpy as np
resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
src, arrays = sys.argv[1], sys.argv[2]
ns = {{}}
exec(compile(open(src).read(), "generated_feature", "exec"), ns)
data = np.load(arrays)
out = {{}}
for k in sorted({{n.rsplit("__", 1)[0] for n in data.files}}):
    try:
        v = ns["feature"](data[k + "__image"].astype(float), data[k + "__mask"].astype(bool))
        out[k] = float(v)
    except Exception as e:
        out[k] = "error: " + type(e).__name__ + ": " + str(e)[:200]
print(json.dumps(out))
"""


def run(source: str, cases: dict) -> Check:
    """Run the feature on {name: (image, mask)} in a separate process. Values are floats or 'error: ...' strings."""
    with tempfile.TemporaryDirectory() as tmp:
        src, arr, runner = (os.path.join(tmp, f) for f in ("feature.py", "cases.npz", "runner.py"))
        with open(src, "w") as f:
            f.write(source)
        np.savez(arr, **{f"{k}__image": im for k, (im, _) in cases.items()}, **{f"{k}__mask": m for k, (_, m) in cases.items()})
        with open(runner, "w") as f:
            f.write(_RUNNER.format(cpu=CPU_S))
        try:
            p = subprocess.run([sys.executable, "-I", runner, src, arr], capture_output=True, text=True,
                               timeout=TIMEOUT_S, cwd=tmp, env={"PATH": os.environ.get("PATH", "")})
        except subprocess.TimeoutExpired:
            return Check(False, [f"timed out after {TIMEOUT_S} s"])
    if p.returncode != 0:
        return Check(False, [f"process failed: {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else p.returncode}"])
    try:
        values = json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return Check(False, ["no result printed"])
    errors = [f"{k}: {v}" for k, v in values.items() if isinstance(v, str)]
    errors += [f"{k}: not finite ({v})" for k, v in values.items() if not isinstance(v, str) and not np.isfinite(v)]
    return Check(not errors, errors, values)


def synthetic_cases(size: int = 128, seed: int = 0) -> dict:
    """Shapes with known geometry on a speckle-like background; the lesion is darker than its surroundings."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size] - size / 2
    r, th = np.hypot(yy, xx), np.arctan2(yy, xx)
    masks = {"circle": r < 30,
             "ellipse": (xx / 40) ** 2 + (yy / 18) ** 2 < 1,
             "star": r < 22 + 12 * np.cos(7 * th),
             "small_blob": r < 6}
    out = {}
    for k, m in masks.items():
        img = 140 + 25 * rng.standard_normal((size, size))
        img[m] = 60 + 15 * rng.standard_normal(m.sum())
        out[k] = (np.clip(img, 0, 255), m)
    noise = np.clip(128 + 60 * rng.standard_normal((size, size)), 0, 255)
    out["noise_texture"] = (noise, masks["circle"])
    return out


def test(source: str) -> Check:
    """All three gates. A failed check lists every reason, so the LLM can repair its code."""
    c = static_check(source)
    if not c.ok:
        return c
    cases = synthetic_cases()
    a = run(source, cases)
    if not a.ok:
        return a
    b = run(source, cases)
    errors = []
    if b.ok and any(not np.isclose(a.values[k], b.values[k], equal_nan=False) for k in a.values):
        errors.append("not deterministic: two runs gave different values")
    if len({round(v, 9) for v in a.values.values()}) == 1:
        errors.append("same value on every synthetic shape: the feature does not respond to the lesion")
    return Check(not errors, errors, a.values)


def save_candidate(name: str, source: str, request: str, model: str, check: Check) -> str:
    """Store a passing feature with provenance. Status stays 'candidate' until a person reviews it."""
    if not name.isidentifier():
        raise ValueError(f"feature name must be an identifier: {name!r}")
    os.makedirs(GENERATED_DIR, exist_ok=True)
    path = os.path.join(GENERATED_DIR, f"{name}.py")
    with open(path, "w") as f:
        f.write(source)
    meta = dict(name=name, status="candidate", request=request, model=model,
                created=datetime.now().isoformat(timespec="seconds"), synthetic_values=check.values)
    with open(os.path.join(GENERATED_DIR, f"{name}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return path
