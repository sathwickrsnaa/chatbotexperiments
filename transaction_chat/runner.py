"""Internal HTTP endpoint. No host port, model key, or Docker socket."""
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from transaction_chat.config import CSV_PATH, MAX_CODE_CHARS
from transaction_chat.execution import execute

app = FastAPI(title="Transaction analysis runner", docs_url=None, redoc_url=None)
execution_lock = threading.Lock()


class ExecutionRequest(BaseModel):
    code: str = Field(min_length=1, max_length=MAX_CODE_CHARS)


@app.get("/health")
def health():
    if not CSV_PATH.exists():
        raise HTTPException(503, "Dataset not initialized")
    return {"status": "ok"}


@app.post("/execute")
def run(request: ExecutionRequest):
    # Avoid multiple workers overwhelming the container's resource limits.
    if not execution_lock.acquire(blocking=False):
        raise HTTPException(429, "Runner is busy; try again shortly")
    try:
        return execute(request.code)
    finally:
        execution_lock.release()
