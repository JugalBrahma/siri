from .datasets_data import example_inputs
from langsmith import Client

client = Client()
dataset_name = "Agents Loop" 

def upload_datasets():
    # If the dataset already exists, we skip creating it to prevent conflict errors
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Dataset '{dataset_name}' already exists. Skipping creation.")
        return
    except Exception:
        pass # Dataset does not exist, we will create it

    # 1. Ask LangSmith to create a brand new dataset
    dataset = client.create_dataset(dataset_name=dataset_name)
    print(f"Your new dataset ID is: {dataset.id}")

    # 2. Upload the clean examples (question only)
    for question in example_inputs:
        client.create_example(
            inputs={"question": question},
            dataset_id=dataset.id,
        )
    print("Finished uploading examples!")
