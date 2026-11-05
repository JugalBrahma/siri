# main.py
import config.config
import logging
import os
#from evaluate.datasets_evaluate import upload_datasets
from graph_builder.graph import GraphBuilder
from agents.supervisor_agent import superVisorAgent,sub_actionSuperVisorAgent,sub_infoSuperVisorAgent 
from agents.research_agent import research_agent
from agents.weather_agent import weather_agent
from agents.actionagent import action_agent
from agents.guardrail_agent import guardrail_agent
from agents.output_sanitizer import output_sanitizer_agent
from langsmith import evaluate, Client
from langsmith import traceable
from memory.observability import MemoryObserver
from memory.runtime import create_configured_memory_store


def main():
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    memory_user_id = os.getenv("SIRI_USER_ID", "siri_user")
    memory_observer = MemoryObserver()
    memory_store = create_configured_memory_store(user_id=memory_user_id)
    learning_service = None
    if memory_store is not None:
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
    #upload_datasets()
    builder.build()
    user_query = input("Enter the query to search in dataset (e.g. 'theory of relativity'): ")
    
    client = Client()
    dataset_name = "Agents Loop"
    

    @traceable(run_type="chain")
    def target_function(inputs: dict) -> dict:
        """Runs the graph and counts the nodes executed."""
        user_question = inputs["question"]

        # Semantic memory is fetched once here and shared by all agents this turn.
        initial_state = builder.create_turn_state(
            [{"role": "user", "content": user_question}],
        )

        hop_count = 0
        final_state = initial_state
        for state in builder.graph.stream(initial_state, stream_mode="values"):
            final_state = state
            hop_count += 1

        # The first yield is the initial state, not a hop.
        hop_count = max(0, hop_count - 1)

        final_output = ""
        if "messages" in final_state and len(final_state["messages"]) > 0:
            final_output = final_state["messages"][-1].content

        # Runs memory learning in the background.
        if learning_service is not None:
            learning_service.submit_completed_turn(final_state.get("messages", []))

        return {
            "query_input": user_question,
            "output": final_output,
            "hop_count": hop_count
        }


    

    def count_hops(reference_outputs: dict, outputs: dict) -> dict:
        hop_count = outputs.get("hop_count", 0) if isinstance(outputs, dict) else 0
        return {"key": "hop_count", "score": int(hop_count)}

    all_examples = list(client.list_examples(dataset_name=dataset_name))
    specific_examples = [
        ex for ex in all_examples
        if user_query.lower() in str((ex.inputs or {}).get("question", "")).lower()
    ]

    if not specific_examples:
        print(f"No matching dataset example found. Running graph directly on prompt: '{user_query}'...")
        res = target_function({"question": user_query})
        print("\n--- RESULT ---")
        print("Output:", res["output"])
        print("Hops Executed:", res["hop_count"])
        return

    evaluate(
        target_function,
        data=specific_examples,
        evaluators=[count_hops],
        experiment_prefix="Agents Loop"
    )


if __name__ == "__main__":
    main()
