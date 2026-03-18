from langchain_community.tools import StructuredTool
from pydantic import BaseModel,Field

class MultiplyArgs(BaseModel):
    a: int = Field(required= True, description="The first number to multiply")
    b: int = Field(required= True, description="The second number to multiply")


def multiply(a: int, b: int) -> int:
    """Multiply two numbers together and return the result."""
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply,  
    name = "multiply",
    description="Multiplies two numbers together. Use this tool when you need to multiply two numbers and return the result.",
    args_schema= MultiplyArgs
)

result = multiply_tool.invoke({"a": 3, "b": 4})
print(result)