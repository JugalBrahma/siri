import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
DISTILLATION_MODEL_ENV = os.getenv("DISTILLATION_MODEL")

DISTILLATION_MODEL = DISTILLATION_MODEL_ENV or ("gpt-4o-mini" if OPENAI_API_KEY else "llama3-8b-8192")

# OpenAI client specifically for text-embedding-3-large vectors
embedding_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()

# LLM Client for summarization and semantic fact distillation
if OPENAI_API_KEY and not DISTILLATION_MODEL_ENV:
    client = OpenAI(api_key=OPENAI_API_KEY)
elif NVIDIA_API_KEY and any(k in DISTILLATION_MODEL.lower() for k in ["qwen", "nvidia", "deepseek", "nemotron"]):
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
    )
elif GROQ_API_KEY:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
    )
elif OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = OpenAI()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "aisaftey")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "1024"))

pinecone_index = None
if PINECONE_API_KEY:
    try:
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Check if index exists; create if missing
        if not pc.has_index(PINECONE_INDEX_NAME):
            print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating serverless index...")
            # Siri creates embeddings itself. A plain vector index with this
            # exact dimension avoids mixing Pinecone integrated embeddings with
            # the 1024-dimension vectors returned by get_embedding().
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=PINECONE_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION,
                ),
            )

        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        print(f"Warning: Failed to initialize Pinecone index '{PINECONE_INDEX_NAME}': {e}")

PINECONE_NAMESPACE = "semantic"


def ensure_namespace_exists(
    index,
    namespace: str = PINECONE_NAMESPACE,
    dimension: int = PINECONE_DIMENSION,
) -> bool:
    """
    Ensure a Pinecone namespace exists inside the given index.
    Pinecone creates namespaces lazily on first upsert, so if the
    namespace is missing we seed a dummy vector and delete it immediately.

    Returns True if the namespace already existed, False if it was just created.
    """
    if index is None:
        print(f"  [NS] No index provided — skipping namespace check.")
        return False

    try:
        stats = index.describe_index_stats()
        ns_map = stats.get("namespaces", {})

        if namespace in ns_map and ns_map[namespace].get("vector_count", 0) > 0:
            print(f"  [NS] Namespace '{namespace}' exists "
                  f"({ns_map[namespace]['vector_count']} vectors).")
            return True

        # Namespace missing or empty — seed it.
        dummy_id = "__ns_init_dummy__"
        print(f"  [NS] Namespace '{namespace}' not found. Creating...")
        index.upsert(
            vectors=[{"id": dummy_id, "values": [0.0] * dimension}],
            namespace=namespace,
        )
        index.delete(ids=[dummy_id], namespace=namespace)
        print(f"  [NS] Namespace '{namespace}' created successfully.")
        return False

    except Exception as e:
        print(f"  [NS] Warning: namespace check failed — {e}")
        return False


# Auto-ensure the semantic namespace exists at startup.
if pinecone_index is not None:
    ensure_namespace_exists(pinecone_index)


def get_embedding(text: str, dimensions: int = PINECONE_DIMENSION) -> list:
    """Generate embedding vector using OpenAI text-embedding-3-large matching Pinecone index dim."""
    if not embedding_client:
        raise ValueError("OpenAI embedding client is not initialized.")
    response = embedding_client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        dimensions=dimensions
    )
    return response.data[0].embedding



