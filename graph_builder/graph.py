from langgraph.graph import StateGraph,END
from state.message_state import State

class GraphBuilder:

    def __init__(self, supervisor, sub_infosupervisor, sub_actionsupervisor, researcher, weather, action, guardrail):
        self.supervisor = supervisor
        self.sub_infosupervisor = sub_infosupervisor 
        self.sub_actionsupervisor = sub_actionsupervisor
        self.researcher = researcher
        self.weather = weather
        self.action = action
        self.guardrail = guardrail
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

        graph.set_entry_point("guardrail")

        graph.add_edge("researcher", "sub_infosupervisor")
        graph.add_edge("weather", "sub_infosupervisor")
        graph.add_edge("action", "sub_actionsupervisor")
        graph.add_edge("supervisor", END)
        self.graph = graph.compile()
        return self.graph
    
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
        
        return self.graph.invoke({"messages": messages})

