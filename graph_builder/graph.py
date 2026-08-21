from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state.message_state import State, create_turn_state
from memory.observability import MemoryObserver
from time import perf_counter

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
        self.graph = None
        

    def build(self):
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

    def run(self, messages: list, config: dict = None) -> dict:
        """
        Run the agent workflow
        
        Args:
            messages: List of message objects
            config: Optional config containing thread_id and callbacks
            
        Returns:
            Final state with results
        """
        if self.graph is None:
            self.build()
        
        turn_config = config or {"configurable": {"thread_id": "default_thread"}}
        return self.graph.invoke(self.create_turn_state(messages), config=turn_config)
