"""Bounded subprocess execution, used only by the runner service."""
import json
import os
import signal
import subprocess
import sys
import tempfile

from transaction_chat.config import EXECUTION_TIMEOUT, MAX_OUTPUT_BYTES
from transaction_chat.policy import validate_code


def execute(code: str, timeout: int = EXECUTION_TIMEOUT):
    try:
        validate_code(code)
    except (ValueError, SyntaxError) as exc:
        return {"ok": False, "output": "", "error": str(exc), "truncated": False}

    # Files bound captured output memory. Worker RLIMIT_FSIZE also bounds disk use.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            [sys.executable, "-m", "transaction_chat.worker"],
            stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
            start_new_session=os.name == "posix",
        )
        timed_out = False
        try:
            process.communicate(json.dumps({"code": code}).encode(), timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.communicate()
        stdout.seek(0)
        stderr.seek(0)
        raw = stdout.read(MAX_OUTPUT_BYTES + 1)
        errors = stderr.read(MAX_OUTPUT_BYTES).decode("utf-8", errors="replace")
        truncated = len(raw) > MAX_OUTPUT_BYTES
        output = raw[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        if timed_out:
            error = f"Execution exceeded {timeout} seconds"
        elif process.returncode:
            error = errors or f"Worker exited with code {process.returncode}"
        elif not output.strip():
            error = "No output. Print the result explicitly."
        else:
            error = ""
        if truncated:
            output += "\n[OUTPUT TRUNCATED: narrow the query or print a labeled sample.]"
        return {"ok": not bool(error), "output": output, "error": error, "truncated": truncated}
