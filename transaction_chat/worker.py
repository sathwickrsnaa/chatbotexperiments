"""One subprocess per calculation. Never import this module into the UI."""
import json
import sys
from decimal import Decimal

from transaction_chat.data import load_transactions
from transaction_chat.policy import validate_code


def main():
    # Linux container limits; Windows local checks exercise the other controls.
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (6, 6))
        resource.setrlimit(resource.RLIMIT_FSIZE, (32_768, 32_768))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except ImportError:
        pass
    request = json.loads(sys.stdin.read())
    code = request["code"]
    validate_code(code)
    import pandas as pd
    import numpy as np
    namespace = {"df": load_transactions(), "pd": pd, "np": np, "Decimal": Decimal}
    exec(compile(code, "<generated-analysis>", "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
