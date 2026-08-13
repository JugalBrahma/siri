from langgraph.graph import END
from langgraph.types import Command
from typing import Literal
from langchain_core.messages import AIMessage
from state.message_state import (
    ActionSupervisorDecision,
    InfoSupervisorDecision,
    MainSupervisorDecision,
    State,
    get_supervisor_state,
    new_supervisor_state,
    record_agent_output,
    semantic_memory_prompt,
    set_routing_plan,
)
from .prompt import system_prompt, sub_info_supervisor_prompt, sub_action_supervisor_prompt
from models.models import llm

def superVisorAgent(state:State)->Command[Literal['sub_infosupervisor','sub_actionsupervisor','output_sanitizer']]:
    print("--- SUPERVISOR AGENT ---")
    main_state = get_supervisor_state(state, "main_supervisor")

    # A completed sub-supervisor result is the only result the main
    # supervisor needs. Do not route the same turn again.
    if main_state["agent_outputs"]:
        final_output = "\n\n".join(main_state["agent_outputs"].values())
        return Command(
            update={
                "messages": [AIMessage(content=final_output, name="supervisor")],
                "main_supervisor": main_state,
                "next": "output_sanitizer",
            },
            goto="output_sanitizer",
        )

    messages = [{
        "role": "system",
        "content": system_prompt + semantic_memory_prompt(main_state["semantic_memory"]),
    },] + state["messages"]
    llm_with_structured_output = llm.with_structured_output(MainSupervisorDecision)
    response = llm_with_structured_output.invoke(messages)
    goto = response.next
    print(f"Decision: {goto}")

    if goto == "FINISH":
        return Command(
            goto="output_sanitizer",
            update={
                "messages": [AIMessage(content=response.reasoning, name="supervisor")],
                "main_supervisor": set_routing_plan(main_state, ["output_sanitizer"]),
                "next": "output_sanitizer",
            },
        )

    updated_main_state = set_routing_plan(main_state, [goto])
    update = {"main_supervisor": updated_main_state, "next": goto}
    if goto == "sub_infosupervisor":
        update["info_supervisor"] = new_supervisor_state(
            main_state["semantic_memory"],
        )
    elif goto == "sub_actionsupervisor":
        update["action_supervisor"] = new_supervisor_state(
            main_state["semantic_memory"],
        )

    return Command(goto=goto, update=update)

def sub_infoSuperVisorAgent(state:State)->Command[Literal['researcher','weather','supervisor']]:
    print("--- SUB INFO SUPERVISOR AGENT ---")
    info_state = get_supervisor_state(state, "info_supervisor")

    # Roll the sub-supervisor's clean result up to the main supervisor. The
    # worker's private context and failed attempts never leave the worker.
    if info_state["agent_outputs"]:
        main_state = get_supervisor_state(state, "main_supervisor")
        latest_result = next(reversed(info_state["agent_outputs"].values()))
        main_state = record_agent_output(
            main_state,
            "sub_infosupervisor",
            latest_result,
        )
        return Command(
            goto="supervisor",
            update={"main_supervisor": main_state, "next": "supervisor"},
        )

    messages = [{
        "role": "system",
        "content": sub_info_supervisor_prompt + semantic_memory_prompt(info_state["semantic_memory"]),
    },] + state["messages"]
    llm_with_structured_output = llm.with_structured_output(InfoSupervisorDecision)
    response = llm_with_structured_output.invoke(messages)
    goto = response.next
    print(f"Decision: {goto}")

    info_state = set_routing_plan(info_state, [goto])
    return Command(
        goto=goto,
        update={"info_supervisor": info_state, "next": goto},
    )

def sub_actionSuperVisorAgent(state:State)->Command[Literal['action','supervisor']]:
    print("--- SUB ACTION SUPERVISOR AGENT ---")
    action_state = get_supervisor_state(state, "action_supervisor")

    if action_state["agent_outputs"]:
        main_state = get_supervisor_state(state, "main_supervisor")
        latest_result = next(reversed(action_state["agent_outputs"].values()))
        main_state = record_agent_output(
            main_state,
            "sub_actionsupervisor",
            latest_result,
        )
        return Command(
            goto="supervisor",
            update={"main_supervisor": main_state, "next": "supervisor"},
        )

    messages = [{
        "role": "system",
        "content": sub_action_supervisor_prompt + semantic_memory_prompt(action_state["semantic_memory"]),
    },] + state["messages"]
    llm_with_structured_output = llm.with_structured_output(ActionSupervisorDecision)
    response = llm_with_structured_output.invoke(messages)
    goto = response.next
    print(f"Decision: {goto}")

    action_state = set_routing_plan(action_state, [goto])
    return Command(
        goto=goto,
        update={"action_supervisor": action_state, "next": goto},
    )
