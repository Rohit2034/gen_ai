from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_EUS2"),
    api_key=os.getenv("AZURE_OPENAI_APIKEY_EUS2"),
    azure_deployment=os.getenv("MODEL_NAME", "gpt-4o"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    temperature=0.7,
)

# Tool 1
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> str:
    """Fetch conversion factor"""
    url = "https://api.fastforex.io/fetch-one"

    response = requests.get(
        url,
        params={
            "from": base_currency,
            "to": target_currency,
            "api_key": "YOUR_KEY"
        }
    )

    return json.dumps(response.json())

# Tool 2
@tool
def convert(amount: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Convert currency"""
    return amount * conversion_rate


llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

# Initial message
messages = [
    HumanMessage(content="Find USD to INR rate and convert 100 USD to INR using tools")
]

# First LLM call
ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

conversion_rate = None

# Tool execution loop
for tool_call in ai_message.tool_calls:

    if tool_call["name"] == "get_conversion_factor":
        tool_output = get_conversion_factor.invoke(tool_call["args"])

        # Add ToolMessage (IMPORTANT)
        messages.append(
            ToolMessage(
                content=tool_output,
                tool_call_id=tool_call["id"]
            )
        )

        # Parse result
        data = json.loads(tool_output)
        conversion_rate = list(data["result"].values())[0]

    elif tool_call["name"] == "convert":
        tool_call["args"]["conversion_rate"] = conversion_rate

        tool_output2 = convert.invoke(tool_call["args"])

        messages.append(
            ToolMessage(
                content=str(tool_output2),
                tool_call_id=tool_call["id"]
            )
        )

# Final LLM response
final_response = llm_with_tools.invoke(messages)

print(final_response.content)