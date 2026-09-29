"""Offline checks: python -m unittest discover -s tests -v."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from transaction_chat.agent import ChatEngine
from transaction_chat.data import generate_transactions, load_transactions
from transaction_chat.execution import execute
from transaction_chat.policy import validate_code
from transaction_chat.tools import create_tools


class ScriptedModel:
    """Offline model to check graph plumbing, not real LLM interpretation."""
    def __init__(self, replies):
        self.replies = iter(replies)
        self.inputs = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.inputs.append(messages)
        return next(self.replies)


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "transactions.csv"
        cls.path.write_text(
            "transaction_id,account_id,timestamp,merchant_name,merchant_category,amount,is_fraud\n"
            "t1,a1,2025-03-10 00:00:00,Acme,Retail,100.00,0\n"
            "t2,a1,2025-03-10 23:59:59,Acme,Retail,83.62,1\n"
            "t3,a2,2025-03-11 00:00:00,Acme,Retail,183.62,0\n"
            "t4,a3,2025-03-10 12:00:00,Other,Dining,9.00,0\n",
            encoding="utf-8",
        )
        cls.frame = load_transactions(cls.path)
        cls.environment = patch.dict(os.environ, {"DATA_DIR": cls.directory.name, "OPENBLAS_NUM_THREADS": "1"})
        cls.environment.start()

    @classmethod
    def tearDownClass(cls):
        cls.environment.stop()
        cls.directory.cleanup()

    def test_seeded_data_is_repeatable_and_supports_grouping(self):
        first = generate_transactions()
        second = generate_transactions()
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 5000)
        self.assertTrue(first.transaction_id.is_unique)
        self.assertLess(first.account_id.nunique(), len(first))
        self.assertLess(first.merchant_name.nunique(), len(first))
        self.assertTrue(first.timestamp.is_monotonic_increasing)
        self.assertTrue(0.01 < first.is_fraud.mean() < 0.05)
        self.assertGreaterEqual(first.loc[first.is_fraud == 1, "amount"].min(), 250)

    def test_money_loading_preserves_cents(self):
        self.assertEqual(self.frame.amount_cents.tolist(), [10000, 8362, 18362, 900])

    def test_daily_tools_exclude_next_midnight(self):
        tools = {tool.name: tool for tool in create_tools(self.frame)}
        arguments = {"merchant_name": "acme", "day": "2025-03-10"}
        count = tools["count_merchant_transactions"].invoke(arguments)
        total = tools["sum_merchant_transactions"].invoke(arguments)
        self.assertEqual(count["transaction_count"], 2)
        self.assertEqual(total["total_amount"], "183.62")

    def test_unknown_merchant_and_invalid_date_do_not_become_zero(self):
        tool = create_tools(self.frame)[0]
        with self.assertRaises(ValueError):
            tool.invoke({"merchant_name": "Missing", "day": "2025-03-10"})
        with self.assertRaises(ValueError):
            tool.invoke({"merchant_name": "Acme", "day": "2025-02-30"})

    def test_generated_code_finds_account_totals_exactly(self):
        result = execute("totals = df.groupby('account_id')['amount_cents'].sum()\nprint(totals[totals == 18362].to_dict())")
        self.assertTrue(result["ok"], result)
        self.assertIn("'a1': 18362", result["output"])
        self.assertIn("'a2': 18362", result["output"])
        self.assertNotIn("a3", result["output"])

    def test_worker_is_fresh_between_calls(self):
        first = execute("df.loc[:, 'amount_cents'] = 0\nprint(df.amount_cents.sum())")
        second = execute("print(df.amount_cents.sum())")
        self.assertTrue(first["ok"], first)
        self.assertTrue(second["ok"], second)
        self.assertEqual(first["output"].strip(), "0")
        self.assertEqual(second["output"].strip(), "37624")

    def test_runner_returns_errors_timeout_and_truncation(self):
        self.assertFalse(execute("print(missing_variable)")["ok"])
        self.assertFalse(execute("result = 1 + 1")["ok"])
        timeout = execute("while True:\n    pass", timeout=2)
        self.assertFalse(timeout["ok"])
        self.assertIn("exceeded", timeout["error"])
        output = execute("print('x' * 15000)")
        self.assertTrue(output["ok"])
        self.assertTrue(output["truncated"])

    def test_policy_blocks_common_filesystem_and_code_escape_operations(self):
        for code in ["import os", "open('file')", "df.to_csv('file')", "pd.read_csv('file')", "df.__class__", "getattr(df, '__class__')", "eval('1+1')"]:
            with self.subTest(code=code), self.assertRaises(ValueError):
                validate_code(code)

    def test_clarification_history_tool_results_and_session_isolation(self):
        model = ScriptedModel([
            AIMessage(content="Combined account totals or individual transactions? Which period?"),
            AIMessage(content="", tool_calls=[{"name": "sum_merchant_transactions", "args": {"merchant_name": "Acme", "day": "2025-03-10"}, "id": "call_1", "type": "tool_call"}]),
            AIMessage(content="Acme's total was $183.62."),
            AIMessage(content="A new conversation."),
        ])
        engine = ChatEngine(self.frame, model=model)
        initial = engine.ask("What is Acme's total?", "first")
        self.assertEqual(len(initial["trace"]), 1)
        result = engine.ask("Sum of amounts on 2025-03-10.", "first")
        self.assertIn("183.62", result["answer"])
        self.assertEqual([event["node"] for event in result["trace"]], ["model", "tools", "model"])
        self.assertTrue(any(message.type == "tool" for message in model.inputs[2]))
        self.assertTrue(any(message.content == "What is Acme's total?" for message in model.inputs[1]))
        engine.ask("Hello", "second")
        self.assertFalse(any(message.content == "What is Acme's total?" for message in model.inputs[3]))

    def test_http_runner_health_execution_validation_and_busy_response(self):
        from transaction_chat import runner
        with patch.object(runner, "CSV_PATH", self.path):
            client = TestClient(runner.app)
            self.assertEqual(client.get("/health").status_code, 200)
            result = client.post("/execute", json={"code": "print(len(df))"})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()["output"].strip(), "4")
            self.assertEqual(client.post("/execute", json={"code": ""}).status_code, 422)
            runner.execution_lock.acquire()
            try:
                self.assertEqual(client.post("/execute", json={"code": "print(1)"}).status_code, 429)
            finally:
                runner.execution_lock.release()

    def test_browser_ui_loads_without_an_api_key(self):
        from streamlit.testing.v1 import AppTest
        response = type("HealthyResponse", (), {"raise_for_status": lambda self: None})()
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), patch("transaction_chat.data.CSV_PATH", self.path), patch("transaction_chat.config.CSV_PATH", self.path), patch("httpx.get", return_value=response):
            ui = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
            self.assertEqual(len(ui.exception), 0, ui.exception)
            self.assertTrue(ui.chat_input[0].disabled)
            self.assertEqual(len(ui.dataframe), 1)

    def test_browser_chat_clarification_followup_trace_and_reset(self):
        from streamlit.testing.v1 import AppTest
        model = ScriptedModel([
            AIMessage(content="Which day should I check?"),
            AIMessage(content="", tool_calls=[{"name": "count_merchant_transactions", "args": {"merchant_name": "Acme", "day": "2025-03-10"}, "id": "ui_call", "type": "tool_call"}]),
            AIMessage(content="There were 2 Acme transactions on 2025-03-10."),
        ])
        engine = ChatEngine(self.frame, model=model)
        response = type("HealthyResponse", (), {"raise_for_status": lambda self: None})()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test-key"}), patch("transaction_chat.data.CSV_PATH", self.path), patch("transaction_chat.config.CSV_PATH", self.path), patch("httpx.get", return_value=response), patch("transaction_chat.agent.ChatEngine", return_value=engine):
            ui = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
            ui.chat_input[0].set_value("How many transactions did Acme have?").run()
            self.assertEqual(len(ui.exception), 0, ui.exception)
            self.assertIn("Which day", ui.session_state["messages"][-1]["content"])
            ui.chat_input[0].set_value("2025-03-10").run()
            self.assertEqual(len(ui.exception), 0, ui.exception)
            self.assertIn("2 Acme", ui.session_state["messages"][-1]["content"])
            self.assertTrue(any(e["node"] == "tools" for e in ui.session_state["messages"][-1]["trace"]))
            previous_id = ui.session_state["conversation_id"]
            ui.button[0].click().run()
            self.assertNotEqual(previous_id, ui.session_state["conversation_id"])
            self.assertEqual(ui.session_state["messages"], [])


if __name__ == "__main__":
    unittest.main()
