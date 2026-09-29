"""Explicit model → tools → model graph with memory per browser session."""
import json
from typing import Annotated, TypedDict

import pandas as pd
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from transaction_chat.config import MODEL_NAME
from transaction_chat.prompts import system_prompt
from transaction_chat.tools import create_tools


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    model_calls: int
    tool_calls: int


class ChatEngine:
    def __init__(self, frame: pd.DataFrame, model=None, runner_url=None):
        self.trace = []
        tools = create_tools(frame, **({"runner_url": runner_url} if runner_url else {}))
        self.tools_by_name = {t.name: t for t in tools}
        self.base_model = model if model is not None else ChatOpenAI(
            model=MODEL_NAME, temperature=0, timeout=45, max_retries=1,
        )
        self.model = self.base_model.bind_tools(tools)
        self.instructions = SystemMessage(content=system_prompt(frame))
        builder = StateGraph(State)
        builder.add_node("model", self.call_model)
        builder.add_node("tools", self.call_tools)
        builder.add_edge(START, "model")
        builder.add_conditional_edges("model", self.route)
        builder.add_edge("tools", "model")
        self.graph = builder.compile(checkpointer=InMemorySaver())

    def call_model(self, state):
        count = state.get("model_calls", 0)
        model = self.model
        messages = [self.instructions, *state["messages"]]
        if count >= 3 or state.get("tool_calls", 0) >= 6:
            # Final synthesis call without tool definitions; bounds the agent loop.
            model = self.base_model
            messages.append(SystemMessage(content="Tool budget exhausted. Answer from successful results or explain what could not be computed. Do not request more tools."))
        response = model.invoke(messages)
        self.trace.append({
            "node": "model", "action": "tool_request" if response.tool_calls else "reply",
            "tool_calls": response.tool_calls,
        })
        return {"messages": [response], "model_calls": count + 1}

    def call_tools(self, state):
        outputs = []
        used = state.get("tool_calls", 0)
        for call in state["messages"][-1].tool_calls:
            if used >= 6:
                result = {"ok": False, "error": "Tool budget exhausted"}
            else:
                try:
                    result = self.tools_by_name[call["name"]].invoke(call["args"])
                except Exception as exc:
                    result = {"ok": False, "error": str(exc)}
                used += 1
            self.trace.append({"node": "tools", "name": call["name"], "arguments": call["args"], "result": result})
            outputs.append(ToolMessage(content=json.dumps(result, default=str), tool_call_id=call["id"]))
        return {"messages": outputs, "tool_calls": used}

    @staticmethod
    def route(state):
        return "tools" if state["messages"][-1].tool_calls else END

    def ask(self, question: str, conversation_id: str):
        self.trace = []
        result = self.graph.invoke(
            {"messages": [{"role": "user", "content": question}], "model_calls": 0, "tool_calls": 0},
            config={"configurable": {"thread_id": conversation_id}, "recursion_limit": 12},
        )
        content = result["messages"][-1].content
        if isinstance(content, list):
            content = "\n".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
        return {"answer": content, "trace": list(self.trace)}
