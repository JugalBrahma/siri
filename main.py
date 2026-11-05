# main.py
import config.config
import logging
import os
import uuid
from graph_builder.graph import GraphBuilder, open_sqlite_checkpointer, pending_interrupt
from agents.supervisor_agent import superVisorAgent,sub_actionSuperVisorAgent,sub_infoSuperVisorAgent
from agents.research_agent import research_agent
from agents.weather_agent import weather_agent
from agents.actionagent import action_agent
from agents.guardrail_agent import guardrail_agent
from agents.output_sanitizer import output_sanitizer_agent
from langgraph.types import Command
from langsmith import traceable
from memory.chat_history import ChatHistory, default_db_path
from memory.observability import MemoryObserver
from memory.runtime import create_configured_memory_store


def default_thread_id() -> str:
    """The conversation the CLI continues. One fixed thread until multi-chat exists."""
    return os.getenv("SIRI_USER_ID", "siri_user")


def create_builder(enable_learning: bool = True, persist: bool = True):
    """Build the compiled agent graph. Returns (builder, learning_service).

    Shared by main() and evaluate/run_eval.py so both run the same graph.
    Evaluation passes enable_learning=False and persist=False so test
    questions never reach long-term memory, chat history, or the checkpoint file.
    """
    memory_user_id = os.getenv("SIRI_USER_ID", "siri_user")
    memory_observer = MemoryObserver()
    memory_store = create_configured_memory_store(user_id=memory_user_id)
    learning_service = None
    if enable_learning and memory_store is not None:
        # Lazy import: skip the distillation client when memory isn't configured.
        from memory.learning import MemoryLearningService

        learning_service = MemoryLearningService(
            memory_store=memory_store,
            user_id=memory_user_id,
            observer=memory_observer,
        )

    builder = GraphBuilder(
        guardrail=guardrail_agent,
        supervisor=superVisorAgent,
        sub_actionsupervisor=sub_actionSuperVisorAgent,
        sub_infosupervisor=sub_infoSuperVisorAgent,
        action=action_agent,
        researcher=research_agent,
        weather=weather_agent,
        output_sanitizer=output_sanitizer_agent,
        memory_store=memory_store,
        memory_observer=memory_observer,
        checkpointer=open_sqlite_checkpointer(default_db_path()) if persist else None,
        chat_history=ChatHistory(default_db_path()) if persist else None,
    )
    builder.build()
    return builder, learning_service


def _drive(builder: GraphBuilder, graph_input, config: dict) -> dict:
    """Stream one run (a new turn or a resume) and finish it if it completed."""
    thread_id = config["configurable"]["thread_id"]

    # "updates" mode yields {node_name: update} once per executed node,
    # so `path` lists every hop in order, including repeat visits.
    path: list[str] = []
    error = None
    try:
        for chunk in builder.stream(graph_input, stream_mode="updates", config=config):
            for node in chunk:
                if not node.startswith("__"):
                    path.append(node)
    except Exception as exc:
        # The checkpoint stays on disk; the next startup resumes from here.
        error = f"{type(exc).__name__}: {exc}"

    snapshot = builder.get_state(config)
    interrupt = pending_interrupt(snapshot)
    messages = list(snapshot.values.get("messages", []))
    if error is None and interrupt is None:
        messages = builder.complete_turn(thread_id, messages)

    output = messages[-1].content if messages and interrupt is None else ""
    return {
        "output": output,
        "path": path,
        "hop_count": len(path),
        "blocked": path == ["guardrail"] and error is None,
        "interrupted": interrupt is not None,
        "interrupt": interrupt,
        "error": error,
        "messages": messages,
    }


@traceable(run_type="chain", name="siri_turn")
def run_turn(
    builder: GraphBuilder,
    question: str,
    callbacks: list | None = None,
    thread_id: str | None = None,
) -> dict:
    """Run one user question through the graph and record the hop path.

    With a thread_id the turn continues that conversation (history is loaded
    and the result saved). Without one it runs on a throwaway thread, which is
    what evaluation uses.
    """
    thread_id = thread_id or str(uuid.uuid4())
    run_config = {"configurable": {"thread_id": thread_id}}
    if callbacks:
        run_config["callbacks"] = callbacks

    # Semantic memory is fetched once here and shared by all agents this turn.
    initial_state = builder.prepare_turn(thread_id, [{"role": "user", "content": question}])
    return _drive(builder, initial_state, run_config)


def resume_turn(builder: GraphBuilder, thread_id: str, answer: str) -> dict:
    """Continue a turn paused on ask_human with the user's answer."""
    return _drive(builder, Command(resume=answer), {"configurable": {"thread_id": thread_id}})


def _answer_questions(builder: GraphBuilder, thread_id: str, result: dict) -> dict:
    """While the agent is waiting on ask_human, ask in the terminal and resume."""
    while result["interrupted"]:
        question = result["interrupt"].get("question", "")
        answer = input(f"Siri asks: {question}\n> ")
        result = resume_turn(builder, thread_id, answer)
    return result


def _print_result(result: dict) -> None:
    print("\n--- RESULT ---")
    print("Output:", result["output"])
    print(f"Hops ({result['hop_count']}):", " → ".join(result["path"]) or "(none)")
    if result["error"]:
        print("Error:", result["error"])
        print("(Progress is saved; restart Siri to continue this turn.)")


def main():
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    builder, learning_service = create_builder()
    thread_id = default_thread_id()

    # Finish any turn a crash or shutdown interrupted last time.
    for resumed in builder.resume_unfinished():
        if resumed["status"] == "success":
            print(f"[Resume] Finished an interrupted turn: {resumed['response']}")
            if learning_service is not None:
                learning_service.submit_completed_turn(resumed["messages"])
        elif resumed["status"] == "error":
            print(f"[Resume] Still failing, will retry next start: {resumed['error']}")

    # A question this conversation was waiting on before the app closed.
    waiting = pending_interrupt(builder.get_state({"configurable": {"thread_id": thread_id}}))
    if waiting is not None:
        answer = input(f"Siri was waiting on: {waiting.get('question', '')}\n> ")
        result = _answer_questions(builder, thread_id, resume_turn(builder, thread_id, answer))
    else:
        question = input("Ask Siri: ")
        result = _answer_questions(builder, thread_id, run_turn(builder, question, thread_id=thread_id))

    # Runs memory learning in the background on this turn's messages only.
    if learning_service is not None and not result["error"]:
        learning_service.submit_completed_turn(result["messages"])

    _print_result(result)


if __name__ == "__main__":
    main()
