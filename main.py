# main.py
import config.config
import logging
import os
import uuid
from graph_builder.graph import GraphBuilder
from agents.supervisor_agent import superVisorAgent,sub_actionSuperVisorAgent,sub_infoSuperVisorAgent
from agents.research_agent import research_agent
from agents.weather_agent import weather_agent
from agents.actionagent import action_agent
from agents.guardrail_agent import guardrail_agent
from agents.output_sanitizer import output_sanitizer_agent
from langsmith import traceable
from memory.observability import MemoryObserver
from memory.runtime import create_configured_memory_store


def create_builder(enable_learning: bool = True):
    """Build the compiled agent graph. Returns (builder, learning_service).

    Shared by main() and evaluate/run_eval.py so both run the same graph.
    Evaluation passes enable_learning=False so test questions are never
    written into the user's long-term memory.
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
    )
    builder.build()
    return builder, learning_service


@traceable(run_type="chain", name="siri_turn")
def run_turn(builder: GraphBuilder, question: str, callbacks: list | None = None) -> dict:
    """Run one user question through the graph and record the hop path."""
    # Semantic memory is fetched once here and shared by all agents this turn.
    initial_state = builder.create_turn_state([{"role": "user", "content": question}])

    # The checkpointer requires a thread_id; use a fresh one per turn.
    run_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    if callbacks:
        run_config["callbacks"] = callbacks

    # "updates" mode yields {node_name: update} once per executed node,
    # so `path` lists every hop in order, including repeat visits.
    path: list[str] = []
    interrupted = False
    error = None
    try:
        for chunk in builder.graph.stream(initial_state, config=run_config, stream_mode="updates"):
            for node in chunk:
                if node == "__interrupt__":
                    interrupted = True  # e.g. ask_human paused for input
                elif not node.startswith("__"):
                    path.append(node)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    messages = builder.graph.get_state(run_config).values.get("messages", [])
    output = messages[-1].content if messages else ""

    return {
        "output": output,
        "path": path,
        "hop_count": len(path),
        "blocked": path == ["guardrail"] and error is None,
        "interrupted": interrupted,
        "error": error,
        "messages": messages,
    }


def main():
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    builder, learning_service = create_builder()

    question = input("Ask Siri: ")
    result = run_turn(builder, question)

    # Runs memory learning in the background.
    if learning_service is not None:
        learning_service.submit_completed_turn(result["messages"])

    print("\n--- RESULT ---")
    print("Output:", result["output"])
    print(f"Hops ({result['hop_count']}):", " → ".join(result["path"]) or "(none)")
    if result["error"]:
        print("Error:", result["error"])


if __name__ == "__main__":
    main()
