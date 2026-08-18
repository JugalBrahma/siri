import os
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "siri"

class Config:
    """Configuartion class for agents"""