# Semantic Memory State

## Aim

Give every user turn one consistent, read-only snapshot of reliable semantic memory. The snapshot is fetched before routing starts, then passed from the main supervisor to the selected sub-supervisor and worker. Workers must never fetch long-term memory themselves or expose their private scratchpad to other agents.

## State model

- `WorkerState` is local to one worker execution. It holds the task, the supplied memory snapshot, retry count, latest tool result, and last error. It is not a LangGraph state field and is never returned upward.
- `SupervisorState` is the shared state for one supervisor scope. It contains `routing_plan`, `agent_outputs`, and `semantic_memory`.
- `State` holds separate supervisor scopes: `main_supervisor`, `info_supervisor`, and `action_supervisor`. The main scope creates a fresh sub-supervisor scope when it routes a turn; the semantic-memory string is passed down without another store read.
- `agent_outputs` is an agent-name keyed dictionary. Recording a result replaces that agent's previous value, so only its latest successful output is visible to the supervisor.

## Current progress

- [x] Defined `WorkerState` and `SupervisorState` contracts in `state/message_state.py`.
- [x] Added helpers for fresh supervisor scopes, routing plans, overwrite-by-agent output storage, worker-state construction, and safe prompt rendering.
- [x] Added separate main, information, and action supervisor scopes to the LangGraph state.
- [x] Connected all supervisors and workers to the same turn-scoped semantic-memory snapshot.
- [x] Updated `GraphBuilder` to call `memory_store.format_for_injection()` once before graph execution. A missing or temporarily failing store leaves the profile empty and does not stop the turn.
- [x] Added `PineconeSemanticMemoryStore.format_for_injection()` to turn reliable facts into the profile passed to the graph.
- [x] Kept worker invocation inputs to messages only; worker-local state is not propagated through LangGraph.
- [x] Instantiate the real Pinecone store and JSON metadata repository in `main.py`, then inject it into `GraphBuilder` when the required credentials exist.
- [x] Add post-turn episode summarization, a persisted background queue, and batch distillation that merges durable facts through the store.
- [x] Add unit tests with a fake memory store proving one fetch per turn, scope isolation, overwrite semantics, and no worker scratchpad leakage.
- [x] Add privacy-safe metrics for memory retrieval and distillation: timing, fact counts, profile size, confidence range, action counts, and error type only.

## Next steps

- [ ] Add a database-backed metadata repository before multi-process or multi-user deployment; the current JSON repository is intentionally local and single-process.
- [ ] Add integration tests against a non-production Pinecone index and a mocked OpenAI client.
- [ ] Add a user-facing memory settings flow for review, correction, deletion, and opt-out.
