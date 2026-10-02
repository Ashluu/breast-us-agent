"""P3 orchestrator: a LangGraph StateGraph implementing the ReAct loop over the tools in
agent/tools/imaging_tools.py. See P3_AGENT_NOTES.md "Why LangGraph over plain LangChain
here" for why this is an explicit graph and not an AgentExecutor/create_react_agent
one-liner -- this shape is also the architecture diagram for the paper.

    START -> orchestrator --(tool_calls present)--> tools -+
                ^                                          |
                +------------------------------------------+
                |
                +--(no tool_calls)--> END

`orchestrator` is one LLM turn; `tools` executes every tool call the last AI message asked
for and appends the results as ToolMessages -- that's what lets the loop continue, since the
LLM sees its own tool results on the next turn and decides whether it has enough to call
generate_report, or needs another modality/segmentation call first.
"""

from __future__ import annotations

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import CONFIG
from .tools import TOOLS

SYSTEM_PROMPT = """You are the orchestrator of a small diagnostic-support "AI Board" for \
breast imaging (research pilot). You have four tools: call_us, call_mg, segment_lesion, \
generate_report.

Rules:
- Call call_us or call_mg to get a malignancy probability for an image. Only call a \
modality tool for a modality you were actually given an image path for.
- You may call segment_lesion to get the lesion location, but a case does not require it.
- You MUST finish every case by calling generate_report with a JSON object of whatever \
findings you gathered -- do not write the final report yourself in plain text, the tool \
composes it from a fixed template. You decide the ANALYSIS (which tools, in what order, \
when you have enough); the template decides the WORDING. This split exists specifically to \
avoid the label-leakage risk documented in P3_AGENT_NOTES.md.
- Do not call the same tool with the same input twice.
- Stop calling tools as soon as you have called generate_report.

If the user asks a FOLLOW-UP question about a case you already reported on, answer using
ONLY the numbers/values your own earlier tool calls already produced in this conversation --
quote them back, don't restate them differently, and don't invent a number that was never
returned by a tool. If answering needs information you don't have yet (a modality you never
called, a question about a different case), say so plainly or call the relevant tool -- don't
guess. You are allowed to explain and reason in plain language when answering a follow-up
(unlike generate_report, which is template output); you are not allowed to state a new
diagnostic number that didn't come from a tool call.

BI-RADS: no tool returns a BI-RADS category and you have no access to one. Never state, \
estimate or map a probability to a BI-RADS category, even if asked directly or asked to \
"assume" or "approximate" one. Say that BI-RADS is assigned by a radiologist and that this \
system only outputs a malignancy probability. Never put a birads field into generate_report."""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def _make_llm() -> ChatOllama:
    return ChatOllama(model=CONFIG.ollama_model, base_url=CONFIG.ollama_base_url, temperature=0)


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"


def build_agent():
    """Returns a compiled LangGraph app. Call `.invoke({"messages": [...]})` on it, or
    `.stream(...)` to watch it reason step by step -- run_case.py uses invoke and then
    saves the full message history, which is enough for a per-step trace either way."""
    llm = _make_llm().bind_tools(TOOLS)

    def orchestrator_node(state: AgentState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.set_entry_point("orchestrator")
    graph.add_conditional_edges("orchestrator", _should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "orchestrator")

    return graph.compile()


def run_one_case(image_path: str, modality: str) -> dict:
    app = build_agent()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"New case. Modality: {modality}. Image path: {image_path}. "
            f"Assess it and produce a report."
        )),
    ]
    return app.invoke({"messages": messages})


def ask_followup(prior_result: dict, question: str) -> dict:
    """Continue an already-finished case with a follow-up question, grounded in that case's
    own tool-call history -- not a fresh, unrelated chat. `prior_result` is whatever
    `run_one_case` (or a previous `ask_followup`) returned; its full message history
    (including every ToolMessage with the actual numbers) is what the orchestrator sees
    again, so "why do you say that" can reference real findings instead of guessing.

    Re-invokes the SAME graph shape -- no new node/edge, no checkpointer needed, because
    the whole "session" is just this messages list; appending one HumanMessage and
    re-running is a normal ReAct turn, it can still call a tool (e.g. call_mg for a second
    modality) if the question calls for it, per the follow-up rules in SYSTEM_PROMPT.

    Example:
        result = run_one_case("data/bus_bra/images/bus_0314-r.png", "us")
        result = ask_followup(result, "Neden malignant-leaning dedin, hangi sayilara gore?")
        print(result["messages"][-1].content)
    """
    app = build_agent()
    messages = list(prior_result["messages"]) + [HumanMessage(content=question)]
    return app.invoke({"messages": messages})
