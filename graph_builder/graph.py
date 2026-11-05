import logging
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union
from time import perf_counter
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state.message_state import State, create_turn_state, current_turn_messages, latest_user_task
from memory.observability import MemoryObserver
from services.text import message_text

StreamMode = Literal['checkpoints', 'custom', 'debug', 'messages', 'tasks', 'updates', 'values']

LOGGER = logging.getLogger("siri.graph")

# Write each node's checkpoint to disk before the next node starts, so a crash
# loses at most the node that was running.
DURABILITY = "sync"


def open_sqlite_checkpointer(db_path: str | Path):
    """A LangGraph checkpointer backed by a local SQLite file."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")  # lets chat history and checkpoints share the file
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def pending_interrupt(snapshot: Any) -> Optional[dict]:
    """The first ask_human payload a paused run is waiting on, else None."""
    for task in getattr(snapshot, "tasks", None) or ():
        for item in getattr(task, "interrupts", None) or ():
            value = getattr(item, "value", item)
            return value if isinstance(value, dict) else {"type": "ask_human", "question": str(value)}
    return None


def _thread_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}

class GraphBuilder:

    def __init__(
        self,
        supervisor,
        sub_infosupervisor,
        sub_actionsupervisor,
        researcher,
        weather,
        action,
        guardrail,
        output_sanitizer,
        memory_store=None,
        memory_observer=None,
        checkpointer=None,
        chat_history=None,
        ask_user=None,
        history_limit: int = 10,
    ):
        self.supervisor = supervisor
        self.sub_infosupervisor = sub_infosupervisor 
        self.sub_actionsupervisor = sub_actionsupervisor
        self.researcher = researcher
        self.weather = weather
        self.action = action
        self.guardrail = guardrail
        self.output_sanitizer = output_sanitizer
        # Supervisor human-in-the-loop: pauses for the user's answer.
        if ask_user is None:
            from agents.clarify import ask_user_node as ask_user
        self.ask_user = ask_user
        # The store is optional while persistence infrastructure is being
        # wired. When present, it is read exactly once by create_turn_state.
        self.memory_store = memory_store
        self.memory_observer = memory_observer or MemoryObserver()
        # Per-node save points. In-memory by default; pass open_sqlite_checkpointer()
        # so an unfinished turn survives a crash or restart.
        self.checkpointer = checkpointer if checkpointer is not None else MemorySaver()
        # Optional ChatHistory: user messages + final answers only.
        self.chat_history = chat_history
        self.history_limit = history_limit
        self.graph: Any = None
        

    def build(self):
        # pyrefly: ignore[bad-specialization]
        graph = StateGraph(State)
        graph.add_node("guardrail", self.guardrail)
        graph.add_node("supervisor", self.supervisor)
        graph.add_node("sub_infosupervisor", self.sub_infosupervisor)
        graph.add_node("sub_actionsupervisor", self.sub_actionsupervisor)
        graph.add_node("action", self.action)
        graph.add_node("researcher", self.researcher)
        graph.add_node("weather", self.weather)
        graph.add_node("output_sanitizer", self.output_sanitizer)
        graph.add_node("ask_user", self.ask_user)

        graph.set_entry_point("guardrail")

        graph.add_edge("researcher", "sub_infosupervisor")
        graph.add_edge("weather", "sub_infosupervisor")
        graph.add_edge("action", "sub_actionsupervisor")
        graph.add_edge("output_sanitizer", END)
        self.graph = graph.compile(checkpointer=self.checkpointer)
        return self.graph
    
    def _ensure_graph(self):
        """Compile the graph on first use and return it."""
        if self.graph is None:
            self.graph = self.build()
            if self.graph is None:
                raise RuntimeError("Failed to build the graph: self.build() returned None.")
        return self.graph

    def _fetch_semantic_memory(self) -> str:
        """Fetch a single durable-memory snapshot for the next graph turn."""
        if self.memory_store is None:
            print("[Memory Fetch] Semantic memory store is disabled / unconfigured.")
            return ""

        started_at = perf_counter()
        try:
            profile = self.memory_store.format_for_injection()
            metrics_getter = getattr(self.memory_store, "get_last_retrieval_metrics", None)
            metrics = metrics_getter() if callable(metrics_getter) else {}
            elapsed_ms = (perf_counter() - started_at) * 1000
            self.memory_observer.fetch_succeeded(
                elapsed_ms=elapsed_ms,
                profile=profile,
                metrics=metrics,
            )
            print(f"[Memory Fetch] Retrieved {metrics.get('fact_count', 0)} facts in {round(elapsed_ms, 1)}ms: {profile or '(no facts stored yet)'}")
            return profile
        except Exception as exc:
            # Memory retrieval should not prevent the assistant from handling
            # the current user request. The next turn can retry the fetch.
            elapsed_ms = (perf_counter() - started_at) * 1000
            self.memory_observer.fetch_failed(
                elapsed_ms=elapsed_ms,
                error=exc,
            )
            print(f"[Memory Fetch] Failed after {round(elapsed_ms, 1)}ms: {exc}")
            return ""

    def create_turn_state(self, messages: list) -> State:
        """Build a turn state after fetching semantic memory exactly once."""
        semantic_memory = self._fetch_semantic_memory()
        return create_turn_state(messages, semantic_memory)

    def run(self, messages: list, config: Optional[RunnableConfig] = None) -> dict:
        """
        Run the agent workflow
        
        Args:
            messages: List of message objects
            config: Optional config containing thread_id and callbacks
            
        Returns:
            Final state with results
        """
        graph = self._ensure_graph()
        
        turn_config: RunnableConfig = config or {"configurable": {"thread_id": "default_thread"}}
        return graph.invoke(self.create_turn_state(messages), config=turn_config, durability=DURABILITY)

    def stream(
        self,
        input_data: Any,
        stream_mode: Union[StreamMode, Sequence[StreamMode]] = "values",
        config: Optional[RunnableConfig] = None,
    ):
        """Stream the agent workflow execution with automatic compilation.
        
        Args:
            input_data: Initial state dict or Command object (e.g. Command(resume=...)).
            stream_mode: LangGraph stream mode (default: 'values').
            config: Optional configuration dictionary.
            
        Yields:
            State dicts as nodes in the graph execute.
        """
        graph = self._ensure_graph()
        
        turn_config: RunnableConfig = config or {"configurable": {"thread_id": "default_thread"}}
        return graph.stream(input_data, config=turn_config, stream_mode=stream_mode, durability=DURABILITY)

    def get_state(self, config: Optional[RunnableConfig] = None):
        """Get the current graph state for the given configuration."""
        graph = self._ensure_graph()
        # pyrefly: ignore[bad-argument-type]
        return graph.get_state(config)

    # ------------------------------------------------------------------
    # Turn lifecycle: prepare → stream → complete (or pause on ask_human)
    # ------------------------------------------------------------------

    def prepare_turn(self, thread_id: str, new_messages: list, load_history: bool = True) -> State:
        """Build the input state for a new turn on `thread_id`.

        Any checkpoint left on the thread belongs to a turn that never
        finished (e.g. an ask_human question the user ignored); a new message
        replaces that turn instead of mixing into it. Recent chat history is
        placed before the new messages unless the caller sends its own.
        """
        self._ensure_graph()
        self.checkpointer.delete_thread(thread_id)
        history = []
        if load_history and self.chat_history is not None:
            history = self.chat_history.recent(thread_id, self.history_limit)
        return self.create_turn_state(history + list(new_messages))

    def complete_turn(self, thread_id: str, messages: list) -> list:
        """Save a finished turn to chat history and delete its checkpoint.

        Returns only this turn's messages (from the latest user message on),
        which is what memory learning should see.
        """
        turn = current_turn_messages(list(messages))
        if self.chat_history is not None and len(turn) > 1:
            question = latest_user_task(turn)
            last = turn[-1]
            content = last.get("content", "") if isinstance(last, Mapping) else getattr(last, "content", "")
            answer = message_text(content)
            if question and answer:
                self.chat_history.add_turn(thread_id, question, answer)
        self.checkpointer.delete_thread(thread_id)
        return turn

    def resume_unfinished(self) -> list[dict]:
        """Continue every turn that a crash or shutdown left unfinished.

        Runs once at startup. Each turn restarts from the node that was
        running when it stopped; nodes that already finished are not re-run.
        Turns paused on ask_human are left waiting for the user's answer.
        """
        graph = self._ensure_graph()
        thread_ids = list(dict.fromkeys(
            saved.config["configurable"]["thread_id"] for saved in self.checkpointer.list(None)
        ))
        results = []
        for thread_id in thread_ids:
            config = _thread_config(thread_id)
            snapshot = graph.get_state(config)
            if pending_interrupt(snapshot) is not None:
                continue
            try:
                if snapshot.next:
                    print(f"[Resume] Continuing thread '{thread_id}' at {', '.join(snapshot.next)}")
                    for _ in self.stream(None, stream_mode="updates", config=config):
                        pass
                    snapshot = graph.get_state(config)
                    interrupt = pending_interrupt(snapshot)
                    if interrupt is not None:
                        results.append({"thread_id": thread_id, "status": "interrupted", "interrupt": interrupt})
                        continue
                messages = list(snapshot.values.get("messages", []))
                turn = self.complete_turn(thread_id, messages)
                answer = message_text(getattr(turn[-1], "content", "")) if turn else ""
                results.append({"thread_id": thread_id, "status": "success", "response": answer, "messages": turn})
            except Exception as exc:
                # Leave the checkpoint in place; the next startup tries again.
                LOGGER.warning("Could not resume thread %s: %s", thread_id, exc)
                results.append({"thread_id": thread_id, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return results
