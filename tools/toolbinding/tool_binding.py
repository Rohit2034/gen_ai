from dotenv import load_dotenv
import os

load_dotenv()

# LLM setup
try:
    from langchain_openai import AzureChatOpenAI

    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_EUS2"),
        api_key=os.getenv("AZURE_OPENAI_APIKEY_EUS2"),
        azure_deployment=os.getenv("MODEL_NAME", "gpt-4o"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=0.7,
    )
except Exception:
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    hf_llm = HuggingFacePipeline.from_model_id(
        model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        task='text-generation',
        pipeline_kwargs={"temperature": 0.5, "max_new_tokens": 100}
    )
    llm = ChatHuggingFace(llm=hf_llm)

# Tools
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together and return the result."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together and return the result."""
    return a + b

llm_with_tools = llm.bind_tools([multiply, add], tool_choice="auto")

# Query
messages = [HumanMessage(content="Use tool to multiply 5 and 1000")]

response = llm_with_tools.invoke(messages)
messages.append(response)

# ✅ Safe check
if response.tool_calls:
    tool_call = response.tool_calls[0]

    # Extract args properly
    tool_args = tool_call["args"]

    # Call tool
    result = multiply.invoke(tool_args)

    # Send result back to LLM
    messages.append(
        ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )
    )

    final_response = llm_with_tools.invoke(messages)
    print(final_response.content)

else:
    print("No tool call, direct answer:", response.content)