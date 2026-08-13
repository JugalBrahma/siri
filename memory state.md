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

## Next steps

- [ ] Instantiate and inject the configured Pinecone store plus its metadata repository into `GraphBuilder` in `main.py`.
- [ ] Add the post-turn episode summary and background distillation queue, then write new durable facts through the store.
- [ ] Add unit tests with a fake memory store to prove one fetch per turn, scope isolation, overwrite semantics, and no worker scratchpad leakage.
- [ ] Add observability for memory retrieval failures, profile size, and fact confidence without logging sensitive memory contents.
