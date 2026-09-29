"""Deterministic tools plus the remote Python analysis tool."""
from datetime import datetime

import httpx
import pandas as pd
from langchain_core.tools import tool

from transaction_chat.config import RUNNER_URL


def merchant_rows(frame: pd.DataFrame, merchant_name: str, day: str):
    try:
        start = pd.Timestamp(datetime.strptime(day, "%Y-%m-%d"))
        if start.strftime("%Y-%m-%d") != day:
            raise ValueError()
    except ValueError:
        raise ValueError("day must be a valid date in YYYY-MM-DD format") from None
    merchants = frame["merchant_name"].str.casefold() == merchant_name.strip().casefold()
    if not merchants.any():
        raise ValueError("Unknown merchant. Ask for an exact name or use Python to inspect merchant names.")
    date_matches = (frame.timestamp >= start) & (frame.timestamp < start + pd.Timedelta(days=1))
    return frame.loc[merchants & date_matches]


def create_tools(frame: pd.DataFrame, runner_url: str = RUNNER_URL):
    @tool
    def count_merchant_transactions(merchant_name: str, day: str) -> dict:
        """Count transaction rows for an exact merchant name on one UTC day.
        Use for 'how many transactions'. Name matching ignores capitalization.
        day must be YYYY-MM-DD. Includes both fraud-flagged and unflagged rows.
        """
        rows = merchant_rows(frame, merchant_name, day)
        return {"merchant": merchant_name, "day": day, "transaction_count": len(rows)}

    @tool
    def sum_merchant_transactions(merchant_name: str, day: str) -> dict:
        """Sum transaction amounts for an exact merchant name on one UTC day.
        Use for 'total transaction amount'. day must be YYYY-MM-DD.
        Includes both fraud-flagged and unflagged rows. Uses integer cents.
        """
        rows = merchant_rows(frame, merchant_name, day)
        cents = int(rows.amount_cents.sum())
        return {
            "merchant": merchant_name, "day": day, "transaction_count": len(rows),
            "total_amount": f"{cents // 100}.{cents % 100:02d}", "currency": "USD",
        }

    @tool
    def execute_dataframe_python(code: str) -> dict:
        """Run Python against the full transaction dataframe df in a separate runner.
        Use for account aggregations, matching amounts, categories, and fraud analysis.
        Available: df, pd, np, Decimal. Each execution starts with a fresh dataframe.
        amount_cents is integer money; use it for exact sums and equality comparisons.
        Print your result, calculation scope, count, and any labeled sample.
        Data-analysis imports only. No filesystem, network, private attributes or plots.
        Maximum execution time is 8 seconds; output is limited to 12,000 bytes.
        """
        try:
            with httpx.Client(timeout=12, trust_env=False) as client:
                response = client.post(f"{runner_url.rstrip('/')}/execute", json={"code": code})
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"Python runner unavailable: {type(exc).__name__}", "output": ""}

    return [count_merchant_transactions, sum_merchant_transactions, execute_dataframe_python]
