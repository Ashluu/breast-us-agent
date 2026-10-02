# P3 agent layer

Design rationale, constraints and decisions live in `../P3_AGENT_NOTES.md` -- this file is
just "how to run it".

## Setup (once, in the Mac's OWN Terminal.app -- natively, NOT through the Claude device-bridge shell)

```
pip install -r requirements.txt
ollama pull hf.co/oamazonasgabriel/lfm2.5-2.6b   # or whatever tag `ollama list` shows after pulling
```

Fix `agent/config.py` if the checkpoint paths, manifest names, or the pulled model's exact
Ollama tag differ from what's there now.

## Run

```
python -m agent.run_case --n 5          # 5 US validation cases -> agent/traces/*.json
python -m agent.run_case --n 5 --modality mg
```

## Follow-up questions on a finished case

```python
from agent.graph import run_one_case, ask_followup

result = run_one_case("data/bus_bra/images/bus_0314-r.png", "us")
result = ask_followup(result, "Neden malignant-leaning dedin, hangi sayilara gore?")
print(result["messages"][-1].content)
```

Grounded in that case's own tool-call history -- the orchestrator quotes back numbers it
already computed, it does not invent new ones. See P3_AGENT_NOTES.md "Follow-up questions
on a finished case" for why this exists.

## Closed loop over views (after CLARITY, arXiv:2512.08029)

```
python -m agent.loop --n 5              # LLM proposes views, frozen probes score them -> agent/traces/loop_*.json
python -m agent.loop --eval --llm-n 60  # validation table: single views vs the loop
```

Propose, score, feed back, refine, for at most 3 rounds. The actions are views of the
same image (full frame, lesion crops, box geometry), not treatments, and the loop
combines every scored view instead of selecting one. Mapping, limits and results:
P3_AGENT_NOTES.md "Closed loop over views". In the UI it is the second button,
"Assess with closed loop over views".

Logic checks that need no checkpoint, data or Ollama: `python -m unittest tests.test_agent_loop`.

## Layout

- `loop.py` -- the propose/score/feedback loop, its view scorer and its validation table.

- `config.py` -- the only knobs (checkpoint/manifest paths, model name).
- `tools/imaging_tools.py` -- call_us, call_mg, segment_lesion, generate_report.
- `graph.py` -- the LangGraph StateGraph (the ReAct loop itself).
- `run_case.py` -- CLI: sample cases from the existing validation split, run the graph, save traces.
- `traces/` -- per-case JSON output, gitignored data would go under `data/`/`checkpoints/` as
  usual but traces are small and are the paper's raw material, so they stay tracked here.
