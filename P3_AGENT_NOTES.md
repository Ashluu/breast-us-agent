# P3 — Agentic Orchestration Layer: Design & Decision Log

> **Phase:** P3 (agentic orchestration) — the "AI Board" layer sitting on top of the P1
> per-modality encoders and (eventually) the P2 cross-modal alignment. Also doubles as the
> project for the Agentic AI course (LangChain-based).
> **Status (2026-09-20):** first code written -- `agent/config.py`, `agent/tools/imaging_tools.py`
> (call_us/call_mg/segment_lesion/generate_report), `agent/graph.py` (the StateGraph),
> `agent/run_case.py` (CLI + trace saving). NOT yet run end to end: needs the real US
> checkpoint copied onto a GPU machine and Ollama + the LFM2.5-2.6B model pulled there
> (see `agent/README.md`). Syntax-checked only (`py_compile`) on this Mac, which has neither
> torch nor Ollama.
> **Deliverable for the course:** written report/paper, not a polished codebase — so this
> phase optimizes for "produces a defensible evaluation + traces to write about", not for
> production code quality.

---

## Why this phase exists

Advisor's stated research interest (his lab page, "AI Agents for Diagnosis and Prognosis")
describes exactly the agentic layer already scoped in `breast_ssl_data_plan.md`:
modality-specific models called as tools by a ReAct-style orchestrator ("AI Board"), producing
patient-specific decisions instead of a single frozen classifier. This phase is that layer,
scaled down to what's gradable in one course term.

## Hard constraints for this phase

- **Local-only inference.** No API budget. Corrected 2026-09-20: there is no local 4090
  box -- real compute is UHEM's A100 (Altay, a100q queue) and this MacBook (M5 Pro, Apple
  Silicon). For a 2.6B orchestrator + a ViT-B encoder, the Mac is actually the better fit:
  Ollama + GGUF/MLX are Apple-Silicon-optimized, no HPC queue wait, and `pip install torch`
  on native macOS gets a small CPU/MPS wheel (not the huge CUDA-bundled Linux one). UHEM A100
  stays the fallback for anything heavier (e.g. if P1/P2 retraining ever needs to happen
  alongside this). Everything here must run in the Mac's OWN Terminal.app, natively --
  NOT through this Claude session's device-bridge shell, which is a disposable, disk-capped
  (~10GB), GPU-less Linux sandbox regardless of the host being a Mac (confirmed 2026-09-20
  the hard way: a plain-Linux `pip install torch` there pulls a multi-GB CUDA dependency
  stack for a GPU that was never there, and blew the sandbox's disk quota).
- **Framework: LangGraph** — decided 2026-09-20, over plain LangChain `AgentExecutor`
  chains. LangGraph is a LangChain-org project (still satisfies a "LangChain" course
  requirement in spirit; confirm against the actual syllabus/rubric once available), and is
  LangChain's own recommended path for anything with loops or branching — see rationale below.
- **No RL this phase.** The full plan calls for RL-based active modality acquisition
  (see `breast_ssl_data_plan.md`); training that is out of scope for a one-term course deliverable. Here
  the orchestrator's own reasoning decides whether to call another modality tool — a stated,
  explicit simplification, not an oversight. RL active acquisition stays a P4/future-work item.
- **No freeform LLM-generated report text.** `generate_report` must compose from the existing
  categorical BI-RADS-style fields (LesionEdge, LesionBoundary, EchoCharacteristics, ...),
  the same fields the text component already uses post-BUS-CoT-leakage-rejection (see
  `DATA_BUS_COT.md`: `reasoning_response` leaked the histopathology label
  in 100% of 2,403 records). Letting the orchestrator LLM free-write a report reopens exactly
  that leakage risk one level up the stack, so it's templated output only.

## Why LangGraph over plain LangChain here

The ReAct loop this layer needs (call a modality tool -> observe result -> decide whether
another modality/segmentation call is needed -> loop or stop -> generate_report) is a graph
with a cycle and a conditional exit, not a straight-line chain. Plain LangChain's
`AgentExecutor` hides that loop inside itself (no visibility/control over individual steps);
LangGraph makes it an explicit state machine:
- **Native cycles + conditional edges** map directly onto "call another modality or stop" —
  exactly the decision an RL policy would eventually make (P4). Building
  it as an explicit graph node now means swapping in a learned policy later is a node
  replacement, not a rewrite.
- **`langgraph.prebuilt.create_react_agent`** gives a ready ReAct implementation to start from
  (works directly with the same LangChain-style tools/models already planned in the tool
  interface contract below — nothing there changes because of this choice).
- **Built-in state/checkpointing** — the per-case tool-call sequence + intermediate results
  needed for `agent/traces/` and the paper's case-study section come out of the graph's own
  state history, instead of hand-rolled logging around an `AgentExecutor` call.

Trade-off, noted honestly: slightly more upfront structure (thinking in nodes/edges/state)
than a one-line `AgentExecutor`, but for a loop-with-a-stopping-condition this pays for itself
almost immediately rather than being over-engineering.

## Orchestrator model: LiquidAI LFM2.5-2.6B

Chosen for the local-only constraint. Verified specs (HF model card + Liquid docs, 2026-09-20):
- 2.69B params (22 double-gated short-conv blocks + 8 GQA layers), hybrid architecture built
  for on-device/agentic deployment.
- 128K context window.
- **Native tool-calling support** via `<|tool_call_start|>`/`<|tool_call_end|>` special tokens
  (Pythonic or JSON call format) — trained with agentic RL inside popular agent frameworks,
  i.e. this is explicitly what it's for, not a generic chat model pressed into tool use.
- Runs under 2.5GB memory even on CPU; trivially fits the Mac's unified memory with enormous headroom.
  GPU throughput reported in the thousands of tokens/sec at concurrency — latency is not a
  concern for a ReAct loop with a handful of tool calls per case.
- GGUF/ONNX/MLX builds exist; a community Ollama build already exists
  (`ollama.com/oamazonasgabriel/lfm2.5-2.6b`), which is the simplest path to a LangChain-
  compatible local endpoint: `ChatOllama(...).bind_tools([...])`. Fallback: `llama.cpp`
  server's OpenAI-compatible endpoint + `ChatOpenAI(base_url=...)`.

**Why small is actually correct here, not a compromise:** the orchestrator's job is tool
routing (which modality to call, when to stop, how to assemble the templated report) — the
actual diagnostic signal comes from the P1/P2 encoders and probes, not from the LLM reasoning
about pixels. A 2.6B routing/composition model is proportionate to that job, and "on-device
agentic system" is a stronger fit for the healthcare-deployment motivation paragraph than
"we called a 70B model" would be.

Open TODO: confirm the course's LangChain version actually supports `.bind_tools()`
against a local Ollama/OpenAI-compatible model (older LangChain releases vary here).

**Setup note (2026-09-20/21, on the Mac's own Terminal, not the Claude device-bridge):**
pulled `oamazonasgabriel/lfm2.5-2.6b:q4_k_m-8gbGPU` via Ollama — the `hf.co/...` tag guessed
earlier in this file does not exist, this is the real one. 🔴 Found and fixed a real bug:
that community build's Modelfile sets `stop` on the bare strings `tool_call_start` /
`tool_call_end` (not the full `<|tool_call_start|>` special-token spelling) — Ollama cuts
generation the instant it sees that substring, i.e. the model's tool-calling output token is
itself the stop condition, so `tool_calls` always came back empty no matter what the model
"wanted" to call. Fixed by deriving a local model with that stop condition removed:

    printf 'FROM oamazonasgabriel/lfm2.5-2.6b:q4_k_m-8gbGPU\nPARAMETER stop <|im_end|>\n' > Modelfile
    ollama create lfm2.5-2.6b-agent -f Modelfile

`agent/config.py`'s `ollama_model` now points at `lfm2.5-2.6b-agent`, not the base pulled tag
— don't revert that without re-checking this stop-token issue. Still open: `agent/traces/`
is empty as of this note, so `python -m agent.run_case` has not yet completed a full
end-to-end case with this fixed model — that's the next thing to actually confirm.

## Tool interface contract

Thin wrappers around existing `bjepa` components — no new modeling code, just callable
signatures the LangChain agent can bind to:

- `call_us(image_path) -> {prob_malignant, embedding, confidence}` — wraps the US I-JEPA
  encoder + frozen probe.
- `call_mg(image_path) -> {prob_malignant, embedding, confidence}` — same, mammography side,
  once that encoder is at parity with the US one.
- `segment_lesion(image_path) -> {bbox, mask_area}` — reuses the same mask logic that backs
  `crop_to_lesion` in `bjepa/transforms.py`; mask stays internal to this tool's return value,
  same leakage-safety argument as the ROI-crop work (mask never becomes model input, only a
  bbox/area summary).
- `generate_report(findings: dict) -> str` — templated composition from categorical fields
  only (see constraint above).

## Follow-up questions on a finished case

Added 2026-09-21, direct response to the advisor's own framing ("interactive framework...
in collaboration with clinicians") -- `run_one_case` alone is one-shot (case in ->
`generate_report` -> stop), which doesn't demonstrate that. `agent/graph.py` now also has
`ask_followup(prior_result, question)`: re-invokes the same graph with the prior case's full
message history plus one new question. The orchestrator can explain/reason in plain language
here (unlike `generate_report`, which stays template-only) but is instructed to only quote
numbers its own earlier tool calls actually produced -- never state a new diagnostic figure
that didn't come from a tool. This is a meaningful addition for the paper: it's the
difference between "outputs a report" and "outputs a report AND can be interrogated about
it," which is closer to what an actual AI Board interacting with a clinician needs.

## Closed loop over views

Added 2026-10-02 after the advisor pointed to CLARITY (Ding et al., arXiv:2512.08029): an
LLM policy proposes candidate therapies, a frozen latent world model scores each one, and
the scores are fed back for K=3 rounds. `agent/loop.py` keeps that propose / score / feed
back / refine shape. It cannot keep the content: CLARITY trains on pre- and post-treatment
scans with treatment and survival records, and this project has one image per lesion.

| CLARITY | `agent/loop.py` |
|---|---|
| action: a therapy | action: a view of the same image (`full`, `full_flip`, `roi`, `roi_wide`, `roi_sq`, `box`) |
| Actor: latent world model | frozen I-JEPA encoder + one linear probe per view, fit on BUS-BRA train |
| feedback: survival risk | each view's malignancy probability and the spread between views |
| objective: lowest risk | a stable estimate; all scored views are averaged (mean logit), none is selected |
| constraint set on proposals | known view, available for this image, not scored before; else fixed order |

Selecting the lowest-risk or most confident view would be choosing the answer, which is
why the loop averages instead. The LLM sees probabilities only, never the image or the
label; its one-line reason is stored in the trace and never reaches the report.

**Validation, v3 seed 0, BUS-BRA (n=385), 2026-10-02** (`python -m agent.loop --eval --llm-n 60`):

| | AUROC | views per case |
|---|---|---|
| `full` alone | 0.814 | 1 |
| `roi` alone (best single view) | 0.888 | 1 |
| `box` alone (no pixels) | 0.878 | 1 |
| mean of all 6 views, no loop | 0.907 | 6 |
| loop, fixed order | 0.905 | 3.95 |
| loop, mask-free views only | 0.815 | 2 |
| loop, LLM policy (60 cases) | 0.949 | 3.47 |
| loop, fixed order (same 60 cases) | 0.945 | 3.53 |

`full`, `roi` and `box` reproduce the numbers in the ROI work, so the scorer is wired the
same way. What can and cannot be said:

- The loop reaches about the all-views average with two fewer views per case. That is the
  early stop, not the LLM.
- The LLM policy is not distinguishable from the fixed order (0.949 vs 0.945 on 60 cases, no
  interval computed). Its proposals were valid in 104 of 106 rounds.
- The gain over `full` comes from views placed with the expert mask. Without a mask the
  loop adds nothing (0.815 vs 0.814). This is the same dependency as in the ROI results.
- 0.905 vs 0.888 has no confidence interval yet and is one seed; test was not touched.

Open: paired bootstrap for loop vs `roi`, seeds 1 and 2, a box source that does not need
the expert (caliper plan), wiring the loop into the UI.

## Evaluation plan

Run the ReAct loop over ~15-20 cases sampled from the existing patient-grouped validation
fold (same fold already used for AUROC in `RESULT_BASELINES.md` / `RESULT_US_V2.md` — reuse,
don't resample). Log per case: tool-call sequence, number of calls, final decision, whether it
matched ground truth, and the generated report text. This produces both the quantitative table
(final-decision accuracy/AUROC, avg. tool-calls per case) and the qualitative case-study
traces the paper write-up needs. Traces go in `agent/traces/`.

**Entry point:** `python -m agent.run_case --n 15` (see `agent/README.md`). `run_case.py`
saves raw message-history JSON per case; it does not auto-score against `true_risk` because
`generate_report`'s output is a templated paragraph, not a bare 0/1 — scoring the table by
hand (or with a small follow-up script once a few traces exist) is the next step after a
first real run.

## Paper outline (course deliverable)

1. Motivation — modality-specific systems lack ReAct-style iterative refinement (advisor's
   own framing); the larger breast imaging system this sits inside.
2. Architecture — Board (orchestrator) + tools, diagram.
3. Tool interfaces (above).
4. Orchestration policy — LangChain agent construction, how it decides to call a second
   modality or stop.
5. Case studies — a few full traces.
6. Quantitative results — table above.
7. Limitations — small local LLM, no RL yet, single-institution data, templated (not
   generative) reporting.
8. Future work — RL-based active modality acquisition (full plan), MRI addition.

---
*Log: created 2026-09-20 during initial P3 scoping conversation. Update this file as
decisions change — same convention as `P2_ALIGNMENT_NOTES.md`.*
