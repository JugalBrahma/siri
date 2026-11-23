# Siri

A multi-agent AI assistant. You ask it something in plain language, and a team of specialised agents works out what to do: look up the weather, search the web, create a Word document, read or write a file, open an app. It remembers things about you between sessions.

Built with [LangGraph](https://langchain-ai.github.io/langgraph/) and LangChain. It runs as a terminal app or as a FastAPI backend.

## What it can do

| Ability | Examples |
|---|---|
| **Weather** | "What's the weather in Tokyo?", "And tomorrow?" (OpenWeather, current and forecast) |
| **Web research** | "Find recent news about solid-state batteries" (Tavily search through MCP) |
| **Word documents** | "Create a report on X", add headings, paragraphs and tables, open the file |
| **Files** | Read, overwrite or append to files inside a fixed workspace folder |
| **Open apps** | Chrome, Safari, Firefox, Word, Excel, PowerPoint, VS Code, Notes, Calculator, Terminal |
| **Multi-step tasks** | "Get the weather in Paris and write it into a Word document" |
| **Asks when unsure** | If a detail is missing (a city, a file name), it pauses and asks you, then carries on |
| **Long-term memory** | Learns facts about you from conversations and uses them in later chats |

## How it works

Each request goes through a graph of agents:

```
guardrail → supervisor ─┬→ sub_infosupervisor ─┬→ researcher (web search)
                        │                      └→ weather
                        ├→ sub_actionsupervisor ──→ action (files, Word, apps)
                        ├→ ask_user (pauses to ask you a question)
                        └→ FINISH
                                  ↓
                          output_sanitizer → answer
```

- **guardrail** blocks jailbreaks and dangerous requests (NeMo Guardrails). It also answers plain greetings.
- **supervisor** decides whether the request needs information, an action, both, or a question back to you.
- **sub-supervisors** pick the right worker. Information goes first, then the action worker uses it.
- **output_sanitizer** cleans the final reply and redacts secrets.

Every node's progress is checkpointed to SQLite. If the app crashes or you close it mid-turn, the next start finishes that turn. Chat history is stored in the same database.

### Memory

Siri keeps semantic memory per user. After a turn, a background step distills what is worth remembering into facts and stores them in Pinecone (full records stay in `memory_data/`). At the start of the next turn the relevant facts are fetched and shared with all agents. Memory is optional: without the Pinecone and OpenAI keys the assistant still works, it just doesn't learn.

## Project layout

```
main.py             Terminal entry point
api/                FastAPI backend (chat, streaming, resume, history, memory)
agents/             Guardrail, supervisors, research, weather, action, sanitizer
graph_builder/      LangGraph graph and SQLite checkpointer
tools/              Weather, search, files/apps, Word, ask_human
memory/             Semantic memory, distillation, chat history
mcp_server/         Standalone Tavily MCP server
config/             Env loading, NeMo Guardrails config
evaluate/           LangSmith evaluation suites and datasets
docs/               API spec and setup guides
tests/              Unit tests
```

## Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # then fill in your keys
```

### Environment variables

| Variable | Needed for |
|---|---|
| `OPENROUTER_API_KEY` | The language model (required). Uses OpenRouter. |
| `TAVILY_API_KEY` | Web search |
| `OPENWEATHER_API_KEY` | Weather |
| `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` | Long-term memory (optional, all required together) |
| `LANGSMITH_API_KEY` | Tracing and evaluation (optional) |
| `SIRI_USER_ID` | Identity that memory and the CLI thread are scoped to (default `siri_user`) |
| `SIRI_WORKSPACE` | Folder Siri may read and write (default `~/Documents/siri`) |
| `LOG_LEVEL` | Logging level (default `INFO`) |

`.env.example` also lists Groq and Pinecone region settings.

## Running

**Terminal:**

```bash
python main.py
```

Type a question at the `Ask Siri:` prompt. The result shows the answer and the path the request took through the agents.

**API server:**

```bash
uvicorn api.app:app --reload
```

Interactive docs are at http://localhost:8000/docs.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check |
| `POST /chat` | Send a message, get a reply |
| `POST /chat/stream` | Same, streamed as Server-Sent Events with live agent hops |
| `POST /chat/resume`, `/chat/resume/stream` | Answer a question Siri paused on |
| `GET /chat/status/{user_id}` | Is a turn waiting for an answer? |
| `GET`/`DELETE /chat/history/{user_id}` | Read or clear chat history |
| `GET`/`POST`/`DELETE /memory/{user_id}` and `/facts` | Inspect, add or delete remembered facts |

The full request and response shapes are in [docs/BACKEND_API_SPEC.md](docs/BACKEND_API_SPEC.md).

## Safety

- File access is confined to the workspace folder (`SIRI_WORKSPACE`). Paths outside it are refused.
- Siri can only open apps from a fixed allow-list. It cannot run arbitrary shell commands.
- Requests pass a guardrail first, and replies pass a sanitizer that redacts secrets.

## Tests and evaluation

```bash
python -m pytest tests
```

Evaluation runs on LangSmith with three suites (routing, guardrail, greeting), each driven by a CSV in `evaluate/`:

```bash
python evaluate/run_eval.py                  # all suites, uploaded to LangSmith
python evaluate/run_eval.py --local          # run locally, print a table
python evaluate/run_eval.py --suite guardrail
```

See [docs/LANGSMITH_SETUP_GUIDE.md](docs/LANGSMITH_SETUP_GUIDE.md) and [docs/NEMO_GUARDRAILS_GUIDE.md](docs/NEMO_GUARDRAILS_GUIDE.md) for more.
