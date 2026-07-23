import os
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "siri"

class Config:
    """Configuartion class for agents"""