# Transaction chat

A modular, local Docker chatbot for learning LangGraph tools and generated Python
analysis. Open the Streamlit UI in your browser and ask questions about a persisted,
synthetic transaction dataframe.

## Start on your computer

Install Docker Desktop and start it, using Linux containers. You also need an OpenAI
API key with access to the configured model. An API account is billed separately
from a ChatGPT subscription.

In PowerShell, from this directory:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set:

```dotenv
OPENAI_API_KEY=your_actual_key
OPENAI_MODEL=gpt-4.1-mini
```

Then start:

```powershell
docker compose up --build -d
```

Open **http://localhost:8501**. The first build installs dependencies and generates
5,000 transactions. Check startup with `docker compose ps -a` or
`docker compose logs -f app runner init-data`.

Without a key, you can browse/download the dataset; chat is disabled. An invalid key
or unavailable model is reported when sending a message. Change `.env`, then run
`docker compose up -d` to apply environment changes.

Stop with `docker compose down`. This preserves the transaction volume. To replace
the synthetic dataset after changing its seed, anchor, or row count, deliberately
remove it with `docker compose down -v`, then start again. That command deletes the
demo dataset volume; it does not save chats.

## Try a conversation

The UI shows a merchant/day question using actual names in the generated dataset.

1. Ask: `Are there accounts with the same total amount of $183.62?`
2. The assistant should clarify individual transactions versus combined totals,
   and the time period.
3. Reply: `Combined transaction amounts per account, across the whole dataset.`
4. Open **Graph steps and tool results** to see generated pandas code and its output.

There may be no matching accounts: this is a random dataset, and zero is a valid
result. Account IDs do not identify unique people. You can also ask:

- `Which five merchants have the highest total transaction amount across all data?`
- `Compare average amounts for fraud-flagged and unflagged transactions across all data.`
- `Show five account IDs and their combined transaction totals across all data.`
- Follow up with `How many transactions did the first account have in that period?`

Clarification and semantic tool selection use the model's judgment. The code bounds
execution and validates tool inputs, but does not guarantee perfect interpretation.
Add a question/expected-query evaluation set before relying on answers.

## Modules

```text
app.py                         Browser chat, dataset preview, graph trace
transaction_chat/
  config.py                    Environment settings and limits
  data.py                      Seeded generation and exact cents loading
  prompts.py                   Schema, clarification and calculation instructions
  tools.py                     Daily merchant tools and runner HTTP client
  agent.py                     LangGraph nodes, routing, budgets and memory
  policy.py                    AST checks for common disallowed operations
  execution.py                 Subprocess deadline and bounded output
  worker.py                    Fresh dataframe and generated-code execution
  runner.py                    Internal FastAPI execution endpoint
compose.yaml                   Data initializer, app and runner containers
tests/test_app.py              Offline behavior and integration checks
```

The flow is browser → model → tool → model → answer. If the model asks a
clarification question, that turn ends without a tool call; the next user reply
continues the same conversation. In-memory checkpoints are isolated per browser
session. **New conversation** resets memory. Restarting the app clears all memory;
chat history is not saved to disk.

The graph uses an explicit tool node rather than hiding execution inside an agent
helper. The UI exposes tool names, arguments, Python code, results and graph order;
it does not display private model reasoning. Each turn allows at most six tool
executions, three tool-enabled model calls and one final synthesis call.

## Synthetic data choices

The generator keeps your seven columns, 5,000 records, 3% fraud probability,
merchant categories and amount distributions. Two deliberate changes make analysis
more useful and reproducible:

- Accounts and merchants are drawn from reusable pools (150 accounts, 30 merchants).
  A new random account for every transaction would make account aggregation mostly
  a single-row operation.
- Dates use a fixed configurable anchor, and UUIDs are seeded. Your original
  `uuid4()` and `end_date="now"` would vary despite random seeds.

Default period: approximately March 1–31, 2025, with UTC semantics. The sidebar shows
the actual minimum and maximum timestamps. Source amounts are stored with two
decimal places. Loading derives `amount_cents` for exact sums and comparisons;
it is an extra calculation column, not a change to the CSV's seven columns.
Amounts are treated as demonstration USD, and fraud flags are synthetic labels.

The CSV is created once in a named Docker volume. Both app and runner mount the
same snapshot read-only, preventing calculation results from changing the source.
Each Python call loads it into a new subprocess; variables and dataframe changes
are discarded after that call.

## Execution boundaries

Generated Python runs in a separate, non-root runner container. It receives no
OpenAI key, has no host port or Docker socket, has a read-only root filesystem and
dataset, a small temporary filesystem, dropped capabilities and resource limits.
The internal Docker network provides no ordinary external network route for the
runner. It can still reach services on that internal network, including the app.
The app alone has an external network connection for model requests.

Each worker has an eight-second wall deadline, a six-second Linux CPU limit, and
bounded captured output. AST checks block common filesystem, network-import and
introspection operations. **These checks are not a complete Python sandbox.** The
container separates execution from the notebook and host, but is not a guarantee
against malicious code or container escape. This is a localhost learning app for
your own synthetic data, not a multi-user public code-execution service.

Hosting locally does not make model inference local: OpenAI receives conversation
messages, schema context and tool results, which can include sampled rows. The full
CSV is not automatically attached. Keep `.env` private; it is excluded from Git and
Docker build context.

## Development and checks without Docker

Use Python 3.12 or later:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c constraints.txt -r requirements.txt -r requirements-runner.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests use temporary fixture data and scripted model replies; they do not make
paid model calls or verify real LLM interpretation. Runner integration checks
execute actual subprocess calculations, including grouped currency matches,
timeouts, errors, output truncation and fresh data between calls. Linux-only limits
and Docker network/filesystem controls require running the containers.

`constraints.txt` pins the dependency versions used for local verification. Both
Docker images use these constraints, while installing only their own requirements.
Local verification used Python 3.13 on Windows; the images use Python 3.12 on Linux.
Docker startup and live model calls need verification on a machine with Docker and
a configured API key.

For optional local development, generate data with `python -m transaction_chat.data`,
start `uvicorn transaction_chat.runner:app --host 127.0.0.1 --port 8000` in one terminal,
and `streamlit run app.py --server.address=127.0.0.1` in another. Set `RUNNER_URL` to
`http://127.0.0.1:8000` and `OPENAI_API_KEY` in the app terminal. Local development
does not provide the Docker execution boundaries. These commands do not read `.env`
automatically; Docker Compose does.
