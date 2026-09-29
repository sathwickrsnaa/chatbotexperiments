import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CSV_PATH = DATA_DIR / "transactions.csv"
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
RUNNER_URL = os.getenv("RUNNER_URL", "http://runner:8000")
EXECUTION_TIMEOUT = 8
MAX_OUTPUT_BYTES = 12_000
MAX_CODE_CHARS = 8_000
