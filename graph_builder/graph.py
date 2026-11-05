from typing import Any, Literal, Optional, Sequence, Union
from time import perf_counter
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state.message_state import State, create_turn_state
from memory.observability import MemoryObserver

StreamMode = Literal['checkpoints', 'custom', 'debug', 'messages', 'tasks', 'updates', 'values']

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
    ):
        self.supervisor = supervisor
        self.sub_infosupervisor = sub_infosupervisor 
        self.sub_actionsupervisor = sub_actionsupervisor
        self.researcher = researcher
        self.weather = weather
        self.action = action
        self.guardrail = guardrail
        self.output_sanitizer = output_sanitizer
        # The store is optional while persistence infrastructure is being
        # wired. When present, it is read exactly once by create_turn_state.
        self.memory_store = memory_store
        self.memory_observer = memory_observer or MemoryObserver()
        self.checkpointer = checkpointer if checkpointer is not None else MemorySaver()
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
        return graph.invoke(self.create_turn_state(messages), config=turn_config)

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
        return graph.stream(input_data, config=turn_config, stream_mode=stream_mode)

    def get_state(self, config: Optional[RunnableConfig] = None):
        """Get the current graph state for the given configuration."""
        graph = self._ensure_graph()
        # pyrefly: ignore[bad-argument-type]
        return graph.get_state(config)
