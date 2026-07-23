import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_nvidia_ai_endpoints")

load_dotenv()
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from langchain_nvidia_ai_endpoints import ChatNVIDIA
model = "openai/gpt-oss-120b"

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=1
)

# model ="minimaxai/minimax-m3"
#model="nvidia/nemotron-3-ultra-550b-a55b"
#model="moonshotai/kimi-k2.6"
#model="deepseek-ai/deepseek-v4-flash"
#model="deepseek-ai/deepseek-v4-pro"
#model="z-ai/glm-5.1"
#model="qwen/qwen3.5-122b-a10b"

# llm = ChatNVIDIA(
#   model=model,
#   api_key=os.environ.get("NVIDIA_API_KEY"),
#   temperature=1,
#   top_p=1,
#   max_completion_tokens=16384,
# )

# lc_messages = [{"role":"user","content":""}]


# response = client.invoke(lc_messages)
# if response.additional_kwargs and "reasoning_content" in response.additional_kwargs:
#   print(response.additional_kwargs["reasoning_content"])
# print(response.content)