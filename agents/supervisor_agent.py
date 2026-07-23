from langgraph.graph import END
from langgraph.types import Command
from typing import Literal
from state.message_state import State, MainSupervisorDecision, InfoSupervisorDecision, ActionSupervisorDecision
from .prompt import system_prompt, sub_info_supervisor_prompt, sub_action_supervisor_prompt
from models.models import llm

def superVisorAgent(state:State)->Command[Literal['sub_infosupervisor','sub_actionsupervisor','output_sanitizer']]:
    print("--- SUPERVISOR AGENT ---")
    messages = [{"role":"system","content":system_prompt},]+ state["messages"]
    llm_with_structured_output = llm.with_structured_output(MainSupervisorDecision)
    response = llm_with_structured_output.invoke(messages)
    goto = response.next
    print(f"Decision: {goto}")

    if goto == "FINISH":
        goto = "output_sanitizer"

    return Command(goto=goto, update={"next": goto})

def sub_infoSuperVisorAgent(state:State)->Command[Literal['researcher','weather']]:
    print("--- SUB INFO SUPERVISOR AGENT ---")
    messages = [{"role":"system","content":sub_info_supervisor_prompt},]+ state["messages"]
    llm_with_structured_output = llm.with_structured_output(InfoSupervisorDecision)
    response = llm_with_structured_output.invoke(messages)
    goto = response.next
    print(f"Decision: {goto}")

    if goto == "supervisor":
        goto="supervisor"

    return Command(goto=goto, update={"next": goto})

def sub_actionSuperVisorAgent(state:State)->Command[Literal['action']]:
    print("--- SUB ACTION SUPERVISOR AGENT ---")
    messages = [{"role":"system","content":sub_action_supervisor_prompt},]+ state["messages"]
    llm_with_structured_output = llm.with_structured_output(ActionSupervisorDecision)
    response = llm_with_structured_output.invoke(messages)
    goto = response.next
    print(f"Decision: {goto}")

    if goto == "supervisor":
        goto="supervisor"

    return Command(goto=goto, update={"next": goto})