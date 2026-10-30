"""Ask the local LLM to write a feature function, test it in the sandbox, and let it repair failures.

    python -m agent.features.generate "margin sharpness: how abruptly brightness changes across the lesion border"
    AGENT_OLLAMA_MODEL=qwen3.5:9b python -m agent.features.generate "..."

Only for features the library does not have (advisor, 30 Oct 2026). At most MAX_TRIES attempts; each failed
attempt sends the sandbox's error list back to the model. A passing function is saved as a candidate.
"""
from __future__ import annotations

import re
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from ..config import CONFIG
from . import sandbox

MAX_TRIES = 3

SYSTEM = f"""You write one Python function that measures a property of a breast ultrasound lesion.

Signature, exactly:
    def feature(image, mask):
`image` is a 2-D float numpy array (grayscale B-mode, 0-255). `mask` is a 2-D bool numpy array of the same
shape, True inside the lesion. Return one Python float.

Rules:
- Allowed imports only: {", ".join(sorted(sandbox.ALLOWED_MODULES))}. No file, network or OS access.
- Nothing at module level except imports and the function. No global variables.
- The value must be finite for any lesion of at least 20 pixels; handle empty regions without dividing by zero.
- Start the function with a one-line docstring saying what the number means and which direction is more suspicious.
Answer with the code only, in one ```python block."""


def _code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


def write_feature(request: str, name: str | None = None, model: str = CONFIG.ollama_model, verbose: bool = True):
    llm = ChatOllama(model=model, base_url=CONFIG.ollama_base_url, temperature=0)
    msgs = [SystemMessage(SYSTEM), HumanMessage(f"Feature to measure: {request}")]
    for attempt in range(1, MAX_TRIES + 1):
        reply = llm.invoke(msgs)
        source = _code(reply.content)
        check = sandbox.test(source)
        if verbose:
            print(f"--- attempt {attempt}: {'PASS' if check.ok else 'FAIL'}\n{source}")
            print("values:" if check.ok else "errors:", check.values if check.ok else check.errors)
        if check.ok:
            path = sandbox.save_candidate(name, source, request, model, check) if name else None
            return dict(ok=True, attempts=attempt, source=source, values=check.values, path=path)
        msgs += [reply, HumanMessage("The sandbox rejected this code:\n- " + "\n- ".join(check.errors[:8]) +
                                     "\nFix it and answer with the full corrected code only.")]
    return dict(ok=False, attempts=MAX_TRIES, source=source, errors=check.errors)


if __name__ == "__main__":
    write_feature(sys.argv[1], name=sys.argv[2] if len(sys.argv) > 2 else None)
