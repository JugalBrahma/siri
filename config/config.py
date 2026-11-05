import os
from dotenv import load_dotenv

load_dotenv(override=True)

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "siri"
