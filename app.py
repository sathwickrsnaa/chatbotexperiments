"""Browser UI. Start with: streamlit run app.py"""
import os
import uuid

import httpx
import streamlit as st

from transaction_chat.agent import ChatEngine
from transaction_chat.config import CSV_PATH, MODEL_NAME, RUNNER_URL
from transaction_chat.data import load_transactions

st.set_page_config(page_title="Transaction chat", page_icon="💬", layout="wide")


@st.cache_data
def dataset():
    return load_transactions()


def render_trace(events):
    if not events:
        return
    with st.expander("Graph steps and tool results"):
        for event in events:
            if event["node"] == "model":
                st.caption("Model → tool request" if event["action"] == "tool_request" else "Model → reply / clarification")
                if event["tool_calls"]:
                    st.json(event["tool_calls"])
            else:
                st.markdown(f"**Tool: `{event['name']}`**")
                arguments = event["arguments"]
                if event["name"] == "execute_dataframe_python":
                    st.code(arguments.get("code", ""), language="python")
                else:
                    st.json(arguments)
                st.json(event["result"])


if not CSV_PATH.exists():
    st.error("Dataset is missing. Start with docker compose up --build, or run python -m transaction_chat.data.")
    st.stop()

frame = dataset()
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.messages = []

has_key = bool(os.getenv("OPENAI_API_KEY", "").strip())
with st.sidebar:
    st.title("Transaction chat")
    st.caption("Local Docker demo · synthetic data")
    st.write(f"Model: `{MODEL_NAME}`")
    st.write(f"{len(frame):,} transactions")
    st.write(f"{frame.account_id.nunique()} accounts · {frame.merchant_name.nunique()} merchants")
    st.write(f"{frame.timestamp.min().date()} → {frame.timestamp.max().date()} (UTC)")
    st.success("API key configured") if has_key else st.warning("Set OPENAI_API_KEY in .env to enable chat")
    try:
        response = httpx.get(f"{RUNNER_URL}/health", timeout=2, trust_env=False)
        response.raise_for_status()
        st.caption("Python runner: ready")
    except httpx.HTTPError:
        st.warning("Python runner unavailable. Advanced analysis will fail until it is running.")
    if st.button("New conversation", width="stretch"):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.pop("engine", None)
        st.rerun()
    st.caption("Chat memory is per browser session and is cleared on restart or New conversation.")

st.title("Ask your transaction data")
st.caption("Ask in plain language. The assistant can clarify, call a tool, and explain the result.")
chat_tab, data_tab, about_tab = st.tabs(["Chat", "Dataset", "How it works"])

with data_tab:
    cols = st.columns(3)
    cols[0].metric("Transactions", f"{len(frame):,}")
    cols[1].metric("Fraud flags", f"{int(frame.is_fraud.sum()):,}")
    cols[2].metric("Accounts", frame.account_id.nunique())
    st.dataframe(frame.head(100), width="stretch", hide_index=True)
    st.caption("Preview: first 100 rows. amount_cents is derived for exact money calculations.")
    st.download_button("Download full synthetic CSV", data=CSV_PATH.read_bytes(), file_name="transactions.csv", mime="text/csv")

with about_tab:
    st.markdown("""
**Each turn:** your message enters the LangGraph model node. The model either asks
for clarification, requests a tool, or answers. Tool results return to the model.

**Tools:** merchant count, merchant amount sum, and generated Python analysis.
Python runs in a separate container against a fresh copy of the persisted dataset.
Open each answer's graph steps to see tool arguments, generated code, and results.

**Data:** 5,000 seeded synthetic transactions, with a 3% fraud probability and
higher amounts for fraud-flagged rows. Accounts and merchants repeat to support
grouped analysis. Amounts are demonstration USD; timestamps represent UTC.

**Connection:** the UI and computation run on your computer. The selected OpenAI
model runs remotely and receives your questions, schema, conversation and tool results.
This is a local learning app; keep the browser endpoint on localhost.
""")

with chat_tab:
    if not st.session_state.messages:
        merchant = str(frame.iloc[0].merchant_name)
        day = frame.iloc[0].timestamp.strftime("%Y-%m-%d")
        st.info(f'Try: How many transactions did "{merchant}" have on {day}?')
        st.caption('Or: "Which accounts have a combined total of $183.62 across all data?" — zero matches is a valid result.')
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_trace(message.get("trace", []))

    question = st.chat_input("Ask about transactions…", disabled=not has_key, max_chars=4000)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                if "engine" not in st.session_state:
                    st.session_state.engine = ChatEngine(frame)
                with st.spinner("Checking the data…"):
                    reply = st.session_state.engine.ask(question, st.session_state.conversation_id)
                st.markdown(reply["answer"])
                render_trace(reply["trace"])
                st.session_state.messages.append({"role": "assistant", "content": reply["answer"], "trace": reply["trace"]})
            except Exception as exc:
                # Discard a partially completed checkpoint after a failed turn.
                st.session_state.pop("engine", None)
                st.session_state.conversation_id = str(uuid.uuid4())
                error = f"This turn failed ({type(exc).__name__}). Check the API key, model access and container logs. Conversation memory was reset; restate your question with its context."
                st.error(error)
                st.session_state.messages.append({"role": "assistant", "content": error})
