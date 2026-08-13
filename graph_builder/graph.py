from langgraph.graph import StateGraph, END
from state.message_state import State, create_turn_state

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
        self.graph = graph.compile()
        return self.graph
    
    def _fetch_semantic_memory(self) -> str:
        """Fetch a single durable-memory snapshot for the next graph turn."""
        if self.memory_store is None:
            return ""

        try:
            return self.memory_store.format_for_injection()
        except Exception as exc:
            # Memory retrieval should not prevent the assistant from handling
            # the current user request. The next turn can retry the fetch.
            print(f"Warning: semantic-memory retrieval failed: {exc}")
            return ""

    def create_turn_state(self, messages: list) -> State:
        """Build a turn state after fetching semantic memory exactly once."""
        semantic_memory = self._fetch_semantic_memory()
        return create_turn_state(messages, semantic_memory)

    def run(self, messages: list) -> dict:
        """
        Run the agent workflow
        
        Args:
            messages: List of message objects
            
        Returns:
            Final state with results
        """
        if self.graph is None:
            self.build()
        
        return self.graph.invoke(self.create_turn_state(messages))

