import os
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_nvidia_ai_endpoints")

load_dotenv(override=True)

# ==============================================================================
# 1. OPENAI (Fast, Reliable Tool Calling, High Rate Limits)
# ------------------------------------------------------------------------------
# To switch to OpenAI, uncomment this section and comment out Groq below:
# ==============================================================================
# model = "gpt-4o-mini"  # Fast, economical, high TPM limit
# model = "gpt-4o"       # Flagship reasoning & multi-step execution
# model = "gpt-5.6-luna"
# llm = ChatOpenAI(
#     model=model,
#     api_key=os.getenv("OPENAI_API_KEY"),
#     reasoning_effort="none",
#     temperature=0,
# )

# ==============================================================================
# Guardrail LLM — lightweight model for NeMo safety rail checks.
# Using a cheaper/faster model here since guardrails only need classification,
# not complex reasoning. Saves ~1k tokens per request.
# ==============================================================================
# guardrail_llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     api_key=os.getenv("OPENAI_API_KEY"),
#     temperature=0,
# )

# ==============================================================================
# 2. OPENROUTER (DeepSeek & Multi-Provider Gateway)
# ==============================================================================
# openrouter_model = "deepseek/deepseek-chat"  # DeepSeek V3 (Chat)
# openrouter_model = "deepseek/deepseek-r1"    # DeepSeek R1 (Reasoning)
# openrouter_model = "deepseek/deepseek-chat:free"
openrouter_model = "deepseek/deepseek-v4.1-flash"
guardrail_llm = ChatOpenAI(
    model=openrouter_model,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)
# To use OpenRouter DeepSeek as the main LLM as well, uncomment below:
llm = ChatOpenAI(
    model=openrouter_model,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# ==============================================================================
# 3. GROQ (Ultra-Fast Inference) [Disabled]
# ==============================================================================
# model = "qwen/qwen3.6-27b"
# model = "llama-3.3-70b-versatile"  # High capacity (30k+ TPM)
# model = "llama-3.1-8b-instant"
# llm = ChatGroq(
#     model=model,
#     temperature=0,
# )
# guardrail_llm = ChatGroq(
#     model=model,
#     temperature=0,
# )
# ==============================================================================
# 3. NVIDIA NIM (Generous Context & Cloud Endpoints)
# ==============================================================================
# model = "meta/llama-3.3-70b-instruct"
# model = "nvidia/nemotron-3-ultra-550b-a55b"
# model = "deepseek-ai/deepseek-v4-pro"
# model = "qwen/qwen3.5-122b-a10b"
# llm = ChatNVIDIA(
#     model=model,
#     api_key=os.getenv("NVIDIA_API_KEY"),
#     temperature=0,
#     max_completion_tokens=16384,
# )