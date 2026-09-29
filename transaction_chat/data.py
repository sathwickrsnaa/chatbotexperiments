"""Generate once, persist, then load the same snapshot in both containers."""
import json
import os
import random
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd

from transaction_chat.config import CSV_PATH, DATA_DIR

CATEGORIES = ["Retail", "Travel", "Dining", "Digital Services", "Groceries"]


def generate_transactions(records=5000, seed=42, anchor="2025-03-31T12:00:00"):
    import numpy as np
    from faker import Faker

    if records < 1:
        raise ValueError("NUM_RECORDS must be positive")
    fake = Faker()
    fake.seed_instance(seed)
    rng = random.Random(seed)
    ids = random.Random(seed + 1)
    amounts = np.random.default_rng(seed)
    end = datetime.fromisoformat(anchor)
    if end.tzinfo is not None:
        raise ValueError("DATA_ANCHOR must be timezone-naive; timestamps represent UTC")
    accounts = [fake.bban() for _ in range(150)]
    merchants = [(fake.company(), rng.choice(CATEGORIES)) for _ in range(30)]
    rows = []
    for _ in range(records):
        fraud = rng.random() < 0.03
        eligible = [m for m in merchants if m[1] in {"Digital Services", "Travel", "Retail"}]
        merchant, category = rng.choice(eligible if fraud else merchants)
        amount = round(float(amounts.uniform(250, 3500) if fraud else amounts.uniform(5, 400)), 2)
        rows.append({
            "transaction_id": str(uuid.UUID(int=ids.getrandbits(128), version=4)),
            "account_id": rng.choice(accounts),
            "timestamp": fake.date_time_between(start_date=end - timedelta(days=30), end_date=end),
            "merchant_name": merchant,
            "merchant_category": category,
            "amount": amount,
            "is_fraud": int(fraud),
        })
    return pd.DataFrame(rows).sort_values("timestamp", kind="stable").reset_index(drop=True)


def load_transactions(path: Path | None = None):
    path = CSV_PATH if path is None else path
    frame = pd.read_csv(path, dtype={"account_id": str, "transaction_id": str, "amount": str})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    decimals = frame["amount"].map(Decimal)
    if any(not x.is_finite() or x * 100 != (x * 100).to_integral_value() for x in decimals):
        raise ValueError("Amounts must be finite and contain at most two decimal places")
    frame["amount_cents"] = decimals.map(lambda x: int(x * 100)).astype("int64")
    frame["amount"] = decimals.map(float)
    return frame


def initialize():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists():
        load_transactions()
        print(f"Using existing dataset: {CSV_PATH}")
        return
    settings = {
        "records": int(os.getenv("NUM_RECORDS", "5000")),
        "seed": int(os.getenv("DATA_SEED", "42")),
        "anchor": os.getenv("DATA_ANCHOR", "2025-03-31T12:00:00"),
    }
    frame = generate_transactions(**settings)
    temporary = CSV_PATH.with_suffix(".tmp")
    frame.to_csv(temporary, index=False, float_format="%.2f")
    temporary.replace(CSV_PATH)
    (DATA_DIR / "metadata.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"Generated {len(frame)} transactions; {int(frame.is_fraud.sum())} fraud flags")


if __name__ == "__main__":
    initialize()
