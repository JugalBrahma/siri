from langgraph.graph import MessagesState,StateGraph,START, END
from pydantic import BaseModel
from typing import Literal

class State(MessagesState):
    next:str

class MainSupervisorDecision(BaseModel):
    reasoning: str
    next: Literal["sub_infosupervisor", "sub_actionsupervisor", "FINISH"]

class InfoSupervisorDecision(BaseModel):
    reasoning: str
    next: Literal["researcher", "weather", "supervisor"]

class ActionSupervisorDecision(BaseModel):
    reasoning: str
    next: Literal["action", "supervisor"]