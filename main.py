# main.py
import config.config
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
def main():
    # Initialize builder with agents
    builder = GraphBuilder(
        guardrail=guardrail_agent,
        supervisor=superVisorAgent,
        sub_actionsupervisor=sub_actionSuperVisorAgent,
        sub_infosupervisor=sub_infoSuperVisorAgent,
        action=action_agent,
        researcher=research_agent,
        weather=weather_agent,
        output_sanitizer=output_sanitizer_agent
    )
    #upload_datasets()
    # Build graph
    builder.build()    
    # Get query from user to search in the LangSmith dataset
    user_query = input("Enter the query to search in dataset (e.g. 'theory of relativity'): ")
    
    client = Client()
    dataset_name = "Agents Loop"
    

    @traceable(run_type="chain")
    def target_function(inputs: dict) -> dict:
        """Runs the RAG graph and extracts the exact nodes executed."""
        
        # 1. Pull the question from LangSmith's dataset input
        user_question = inputs["question"]
        final_output = ""
        
        # 2. Run the graph fully and grab the final state
        # GraphBuilder fetches semantic memory once here, before any routing
        # node runs. All routed agents receive this same turn-scoped snapshot.
        initial_state = builder.create_turn_state(
            [{"role": "user", "content": user_question}],
        )
        
        # Use stream to count the number of nodes executed (hops)
        hop_count = 0
        final_state = initial_state
        for state in builder.graph.stream(initial_state, stream_mode="values"):
            final_state = state
            hop_count += 1
            
        # Subtract 1 because the initial state is yielded first
        hop_count = max(0, hop_count - 1)
        
        # 3. Extract the LLM's final text string
        final_output = ""
        if "messages" in final_state and len(final_state["messages"]) > 0:
            final_output = final_state["messages"][-1].content

        return {
            "query_input": user_question,
            "output": final_output,
            "hop_count": hop_count
        }


    

    def count_hops(reference_outputs: dict, outputs: dict) -> dict:
        hop_count = outputs.get("hop_count", 0) if isinstance(outputs, dict) else 0
        return {"key": "hop_count", "score": int(hop_count)}

    # Fetch all examples and filter for the specific one matching the user's query
    all_examples = list(client.list_examples(dataset_name=dataset_name))
    specific_examples = [
        ex for ex in all_examples 
        if user_query.lower() in str(ex.inputs.get("question", "")).lower()
    ]

    if not specific_examples:
        print(f"ℹ️ No matching dataset example found. Running graph directly on prompt: '{user_query}'...")
        res = target_function({"question": user_query})
        print("\n--- RESULT ---")
        print("Output:", res["output"])
        print("Hops Executed:", res["hop_count"])
        return

    evaluate(
        target_function,
        data=specific_examples,  # Limit to only the first matching example
        evaluators=[count_hops],
        experiment_prefix="Agents Loop"
    )


if __name__ == "__main__":
    main()
