"""One-page UI for the P3 agent: pick a validation case, watch the orchestrator's ReAct
loop step by step (tool calls + results), and read the report.

    python -m agent.app          # then open http://127.0.0.1:7860

Read-only over the same graph run_case.py uses; it writes nothing to agent/traces/.

The report card is built from the TOOLS' own JSON outputs (call_us, segment_lesion), not from
the arguments the LLM passed to generate_report and not from anything the LLM wrote, so the
LLM cannot misreport a number or add diagnostic prose (P3_AGENT_NOTES.md constraint). The
generate_report template is what the chat shows as "the report". Follow-up questions are answered
by the LLM under CHAT_SYSTEM (facts only from tool results, no diagnosis); those replies are
labelled as LLM text, and the manifest label is never put in the LLM's context.

If the US checkpoint is missing, `call_us` is replaced by a DEMO stub (p=0.50, confidence 0)
and the page says so in red. Nothing produced in that mode is a result.
"""

from __future__ import annotations

import html
import json
import os
import re
import time
import traceback

import gradio as gr
import pandas as pd
import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from PIL import Image, ImageDraw

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # manifests use repo-relative paths

from .config import CONFIG
from .graph import SYSTEM_PROMPT, build_agent
from .tools import imaging_tools as tools

STUB = not os.path.exists(CONFIG.us_checkpoint)
if STUB:
    tools._predict_one = lambda image_path, modality: {
        "modality": modality, "prob_malignant": 0.5, "confidence": 0.0}
else:
    # Fitting the probe embeds the whole train split (~30 s on this Mac). Do it now so the
    # first click is not the slow one.
    print("fitting US probe on the train split ...", flush=True)
    _t0 = time.time()
    tools._get_probe("us")
    print(f"probe ready in {time.time() - _t0:.0f}s", flush=True)

_AGENT = build_agent()
def _load_cases() -> pd.DataFrame:
    """BUS-BRA validation (the probe's own fold) plus BUS-CoT *test* images present on this
    machine as an external check. BUS-CoT test is used, not trainval, because BUS-CoT trainval
    is in the v2 SSL pretraining pool (configs/p1_us_ijepa_v2.yaml); its test split is not."""
    bra = pd.read_csv(CONFIG.us_manifest)
    bra = bra[bra["split"] == CONFIG.val_split].assign(source="BUS-BRA validation")
    if not os.path.exists("manifests/bus_cot.csv"):           # not distributed: BUS-CoT licence not confirmed
        return bra[["image_path", "risk", "birads", "source"]].set_index("image_path")
    cot = pd.read_csv("manifests/bus_cot.csv")
    cot = cot[(cot["split"] == "test") & cot["image_path"].map(os.path.exists)]
    cot = cot.assign(source="BUS-CoT test (external)")
    cols = ["image_path", "risk", "birads", "source"]
    return pd.concat([bra[cols], cot[cols]]).set_index("image_path")


_CASES = _load_cases()
_CHOICES = [(f"[{'BRA' if r.source.startswith('BUS-BRA') else 'CoT'}] {os.path.basename(i)}", i)
            for i, r in _CASES.iterrows()]


def _status_md() -> str:
    try:
        r = requests.get(f"{CONFIG.ollama_base_url}/api/tags", timeout=2).json()
        have = {m["name"] for m in r["models"]}
        ok = any(n.split(":")[0] == CONFIG.ollama_model for n in have)
        llm = f"Ollama up, model `{CONFIG.ollama_model}` " + ("found" if ok else "**NOT found**")
    except Exception:
        llm = "Ollama **not reachable**"
    banner = ""
    if STUB:
        banner = (f"<div style='background:#fdecea;color:#b71c1c;padding:10px;border-radius:6px;"
                  f"margin-bottom:8px'><b>DEMO MODE.</b> No US checkpoint at "
                  f"<code>{CONFIG.us_checkpoint}</code>, so <code>call_us</code> returns a fixed "
                  f"placeholder (p=0.50, confidence 0). The LLM, tool routing, segmentation and "
                  f"report are real; the malignancy number is not.</div>")
    enc = "stubbed (demo)" if STUB else f"`{CONFIG.us_checkpoint}` + linear probe fit on the train split"
    return (f"{banner}\n**Orchestrator:** {llm} (runs locally)  \n**US encoder:** {enc}  \n"
            f"**Cases:** {(_CASES.source.str.startswith('BUS-BRA')).sum()} BUS-BRA validation "
            f"(patient-grouped fold, tag `[BRA]`) + {(_CASES.source.str.startswith('BUS-CoT')).sum()} "
            f"BUS-CoT test (external, tag `[CoT]`). The probe was fit on BUS-BRA train only.")


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items())


def _annotate(image_path: str, bbox) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    if bbox:
        d = ImageDraw.Draw(img)
        w = max(2, img.width // 120)
        d.rectangle(bbox, outline=(255, 64, 64), width=w)
    return img


def _card(image_path: str, f: dict) -> str:
    """HTML report card from tool outputs only. Colours are translucent so it reads in both themes."""
    p = f.get("prob_malignant")
    if p is None:
        return "<i>The agent did not obtain a model probability for this case.</i>"
    mal = p >= 0.5
    col = "#c62828" if mal else "#2e7d32"
    label = "Malignant-leaning" if mal else "Benign-leaning"
    pct = round(p * 100, 1)
    rows = [("Modality / source", f"Ultrasound ({_CASES.loc[image_path, 'source']})"),
            ("Model confidence", f"{f.get('confidence', 0):.2f}  <span style='opacity:.6'>(0 = coin flip, 1 = certain)</span>")]
    if f.get("bbox"):
        rows.append(("Lesion bounding box (px)", str(f["bbox"])))
    if f.get("mask_area_fraction") is not None:
        rows.append(("Lesion share of frame", f"{f['mask_area_fraction'] * 100:.1f}%"))
    if f.get("n_views"):
        rows.append(("Views combined (closed loop)", ", ".join(f"{n} {p:.2f}" for n, p in f.get("views", {}).items())))
        rows.append(("Largest disagreement between views",
                     f"{f['view_spread']:.2f}  <span style='opacity:.6'>({'stable' if f.get('stable') else 'unstable'})</span>"))
    if f.get("uses_expert_mask"):
        rows.append(("Lesion box source", "dataset expert mask <span style='opacity:.6'>(not available in deployment)</span>"))
    table = "".join(f"<tr><td style='padding:4px 12px 4px 0;opacity:.7'>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f"""
<div style="border:1px solid rgba(128,128,128,.4);border-radius:12px;padding:18px 20px;max-width:720px">
  <div style="font-size:12px;letter-spacing:.08em;opacity:.65">P3 PILOT REPORT</div>
  <div style="font-size:12px;opacity:.6;margin-bottom:12px">{html.escape(os.path.basename(image_path))}</div>
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">
    <span style="background:{col};color:#fff;padding:4px 12px;border-radius:999px;font-weight:600">{label}</span>
    <span style="font-size:26px;font-weight:700">{pct}%</span>
    <span style="opacity:.65">estimated probability of malignancy</span>
  </div>
  <div style="position:relative;height:12px;border-radius:6px;background:rgba(128,128,128,.25);margin:10px 0 16px">
    <div style="width:{pct}%;height:100%;border-radius:6px;background:{col}"></div>
    <div title="0.5 decision threshold" style="position:absolute;left:50%;top:-4px;width:2px;height:20px;background:rgba(128,128,128,.9)"></div>
  </div>
  <table style="border-collapse:collapse;font-size:14px">{table}</table>
  <div style="margin-top:10px;font-size:12px;opacity:.6">Pilot research system, not a clinical decision.
  Numbers come straight from the tools; the orchestrator LLM wrote none of this text.</div>
</div>"""


CHAT_SYSTEM = """You are the orchestrator of a small diagnostic-support "AI Board" for breast \
ultrasound (research pilot), chatting with a researcher. Tools: call_us, call_mg, \
segment_lesion, generate_report.

Rules:
- Reply in English.
- Facts come ONLY from tool results in this conversation. Never invent findings, BI-RADS \
features, histology, or numbers. If the user asks about the image and the relevant tool has \
not been called yet, call it. If nothing available can answer (for example "why did the model \
say that?"), say so plainly: the encoder outputs a probability and gives no reasons.
- To assess the case call call_us, and segment_lesion when the lesion location or size matters. \
To produce the formal report call generate_report with a JSON object of the findings you \
gathered; never write the report yourself. Do not call the same tool with the same input twice, \
reuse earlier results.
- Never give a definitive diagnosis or treatment advice. This is a research pilot, not a \
clinical tool.
- No tool returns a BI-RADS category. Never state, estimate or map a probability to one, even \
if asked; say BI-RADS is assigned by a radiologist and this system only outputs a malignancy \
probability.
- Keep answers short (2-5 sentences) unless asked for more.
- Only use call_mg for mammography paths. This session's case is ultrasound."""

ASSESS_TEXT = "Assess this case and give me the report."
EXAMPLES = [[ASSESS_TEXT], ["Where is the lesion and how much of the image does it cover?"],
            ["How much can I trust this result?"], ["Is this cancer? What is the definitive diagnosis?"]]


def _template_as_markdown(t: str) -> str:
    """generate_report's fixed template, shown as a chat message. Wording is the template's."""
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    return f"**{lines[0]}**\n\n" + "\n".join(f"- {l}" for l in lines[1:])


def _new_session(case_path: str | None) -> dict:
    where = (f"\nCurrent case: ultrasound. When a tool needs image_path, pass exactly this string, " f"unchanged, with no leading slash: \"{case_path}\"" if case_path else
             "\nNo case is selected yet. Ask the user to pick one before using any tool.")
    return {"case": case_path, "msgs": [SystemMessage(content=CHAT_SYSTEM + where)],
            "findings": {}, "steps": []}


def _ref_md(case_path: str | None, st: dict | None) -> str:
    """Reference label for the USER only, shown inside a collapsed accordion. It is read here
    and nowhere else: it is never part of a message, prompt, tool input or tool output."""
    if not case_path:
        return ""
    row = _CASES.loc[case_path]
    mal = bool(row["risk"])
    txt = (f"**Reference label: {'malignant' if mal else 'benign'}** (BI-RADS {row['birads']})  \n"
           f"_From the manifest ({row['source']}). Never sent to the model._")
    if row["source"].startswith("BUS-CoT"):
        txt += ("  \n_External case: different scanners and a different malignant share than the "
                "BUS-BRA train set the probe was fit on._")
    p = ((st or {}).get("findings") or {}).get("prob_malignant")
    if p is not None:
        agree = (p >= 0.5) == mal
        txt += (f"\n\nThe model said {p * 100:.1f}% malignant, so its prediction "
                f"**{'agrees' if agree else 'disagrees'}** with the reference. "
                f"One case is an anecdote, not evidence.")
    return txt


def on_case(path):
    """Selecting a case starts a fresh conversation about it."""
    return path, [], _new_session(path), "", "", None, gr.Accordion(open=False)


def chat(user_text: str, history: list, case_path: str | None, st: dict | None):
    history = list(history or [])
    if not st or st.get("case") != case_path:
        st = _new_session(case_path)
    if not (user_text or "").strip():
        yield history, "", gr.skip(), gr.skip(), gr.skip(), st
        return
    history.append({"role": "user", "content": user_text})
    msgs = st["msgs"] + [HumanMessage(content=user_text)]
    new, steps, template, final = [], list(st["steps"]), "", ""
    findings = dict(st["findings"])
    t0 = time.time()
    yield history, "", "\n".join(steps) or "_running..._", gr.skip(), gr.skip(), st
    try:
        for step in _AGENT.stream({"messages": msgs}, {"recursion_limit": 12}, stream_mode="updates"):
            for node, upd in step.items():
                for m in upd["messages"]:
                    new.append(m)
                    stamp = f"`{time.time() - t0:4.1f}s`"
                    if getattr(m, "tool_calls", None):
                        for c in m.tool_calls:
                            steps.append(f"{len(steps) + 1}. {stamp} **LLM decides:** call "
                                         f"`{c['name']}({_fmt_args(c['args'])})`")
                    elif node == "tools":
                        steps.append(f"{len(steps) + 1}. {stamp} **tool `{m.name}` returned:** "
                                     f"`{str(m.content)[:200]}`")
                        if m.name in ("call_us", "segment_lesion"):
                            try:
                                got = json.loads(m.content)
                            except (ValueError, TypeError):
                                got = {}
                            # A repeat segment_lesion call with a mangled path returns bbox: null;
                            # never let that erase a box we already have.
                            if m.name == "segment_lesion" and not got.get("bbox") and findings.get("bbox"):
                                got = {}
                            findings.update(got)
                        elif m.name == "generate_report" and not str(m.content).startswith("Error"):
                            template = m.content
                    else:
                        final = m.content
                yield history, "", "\n".join(steps), gr.skip(), gr.skip(), st
    except Exception as e:
        # Roll back this turn: a half-finished tool exchange would poison the next turn's context.
        traceback.print_exc()
        history.append({"role": "assistant",
                        "content": f"⚠️ Agent stopped ({type(e).__name__}). Please retry."})
        yield history, "", "\n".join(steps + [f"**STOPPED:** {type(e).__name__}: {str(e)[:160]}"]), \
            gr.skip(), gr.skip(), st
        return
    st = {**st, "msgs": msgs + new, "findings": findings, "steps": steps}
    if template:
        reply = _template_as_markdown(template)
    else:
        reply = (final or "(empty reply)") + "\n\n<sub>LLM reply, not the report.</sub>"
    history.append({"role": "assistant", "content": reply})
    card, annotated = gr.skip(), gr.skip()
    if case_path and findings.get("prob_malignant") is not None:
        card = _card(case_path, findings)
        if STUB:
            card = ("<div style='color:#b71c1c;font-weight:600;margin-bottom:6px'>DEMO MODE: the "
                    "probability below is a placeholder.</div>") + card
    # The box is independent of the probability: "Where is the lesion?" calls only segment_lesion.
    if case_path and findings.get("bbox"):
        annotated = _annotate(case_path, findings["bbox"])
    yield history, "", "\n".join(steps), card, annotated, st


def assess(history, case_path, st):
    yield from chat(ASSESS_TEXT, history, case_path, st)


_SCORER = None
LOOP_TEXT = "Assess this case with the closed loop over views."


def assess_loop(history, case_path, st):
    """agent/loop.py on the selected case: the LLM proposes views, the frozen probes score
    them, the scores go back for up to three rounds. The chat shows generate_report's
    template; the LLM's one-line reasons appear only in the trace, labelled as such."""
    global _SCORER
    history = list(history or [])
    if not st or st.get("case") != case_path:
        st = _new_session(case_path)
    if not case_path or STUB:
        why = "Pick a case first." if not case_path else "The closed loop needs the US checkpoint (demo mode)."
        yield history + [{"role": "assistant", "content": why}], "", gr.skip(), gr.skip(), gr.skip(), st
        return
    history.append({"role": "user", "content": LOOP_TEXT})
    yield history, "", "_running the closed loop (the first run fits one probe per view, about 30 s each)..._", \
        gr.skip(), gr.skip(), st
    try:
        from . import loop
        if _SCORER is None:
            _SCORER = loop.ViewScorer()
        out = loop.assess(case_path, _SCORER, loop.make_llm_policy())
    except Exception as e:
        traceback.print_exc()
        history.append({"role": "assistant", "content": f"⚠️ Loop stopped ({type(e).__name__}). Please retry."})
        yield history, "", f"**STOPPED:** {type(e).__name__}: {str(e)[:160]}", gr.skip(), gr.skip(), st
        return
    steps = []
    for r in out["rounds"]:
        who = {"llm": "LLM proposes", "fallback": "LLM proposal invalid, fixed order used:"}.get(r["source"], r["source"])
        steps.append(f"**Round {r['round']}.** {who} `{r['proposed'] if r['source'] == 'llm' else r['accepted']}`")
        if r["rejected"]:
            steps.append(f"   - rejected by the constraint check: `{r['rejected']}`")
        steps.append(f"   - scores: `{r['scores']}`, combined {r['aggregate']:.2f}, disagreement {r['spread']:.2f}")
        m = re.search(r'"reason"\s*:\s*"([^"]*)"', r.get("llm_text") or "")
        if m:
            steps.append(f"   - <sub>LLM reason (not part of the report): {html.escape(m.group(1))}</sub>")
    steps.append(f"**Stopped:** {out['stop'].replace('_', ' ')}.")
    findings = {**st["findings"], **out["findings"], "views": out["views"]}
    st = {**st, "findings": findings, "steps": steps,
          "msgs": st["msgs"] + [HumanMessage(content=LOOP_TEXT),
                                AIMessage(content="Closed-loop tool result: " + json.dumps(out["findings"]))]}
    history.append({"role": "assistant", "content": _template_as_markdown(out["report"])})
    yield history, "", "\n".join(steps), _card(case_path, findings), gr.skip(), st


with gr.Blocks(title="P3 agent") as demo:
    gr.Markdown("## P3: agentic orchestration (LFM2.5-2.6B + LangGraph)")
    gr.Markdown(_status_md())
    state = gr.State(None)
    with gr.Row():
        with gr.Column(scale=1):
            case = gr.Dropdown(choices=_CHOICES, label="Case ([BRA] validation, [CoT] external test)",
                               value=None)
            img = gr.Image(label="Selected image", interactive=False, type="filepath")
            go = gr.Button("Assess case and produce report", variant="primary")
            go_loop = gr.Button("Assess with closed loop over views")
            annotated = gr.Image(label="Lesion box from segment_lesion", interactive=False, type="pil")
            with gr.Accordion("Reveal reference label (hidden from the model)", open=False) as ref_acc:
                ref_md = gr.Markdown()
        with gr.Column(scale=2):
            card_html = gr.HTML(label="Report")
            chatbot = gr.Chatbot(label="Chat", height=380)
            msg = gr.Textbox(label="Ask the agent", placeholder="Type a question and press Enter", lines=1)
            gr.Examples(EXAMPLES, inputs=msg, label="Example questions")
            with gr.Accordion("Trace (LLM decisions and tool results)", open=True):
                steps_md = gr.Markdown()
    outs = [chatbot, msg, steps_md, card_html, annotated, state]
    case.change(on_case, case, [img, chatbot, state, steps_md, card_html, annotated, ref_acc]) \
        .then(_ref_md, [case, state], ref_md)
    msg.submit(chat, [msg, chatbot, case, state], outs).then(_ref_md, [case, state], ref_md)
    go.click(assess, [chatbot, case, state], outs).then(_ref_md, [case, state], ref_md)
    go_loop.click(assess_loop, [chatbot, case, state], outs).then(_ref_md, [case, state], ref_md)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
