from langchain_community.tools import BaseTool
from typing import type
from pydantic import BaseModel, Field


class MultiplyArgs(BaseModel):
    a: int = Field(required= True, description="The first number to multiply")
    b: int = Field(required= True, description="The second number to multiply")


#here we get the same functionality as the StructuredTool, but we have more control over how the tool is defined and implemented. We can define our own name, description, and args_schema, and we can implement the _run method to define the logic of the tool. This allows us to create more complex tools that can do more than just call a function. We can also add additional methods or properties to the tool if needed.
#we can also create async tools by implementing the _arun method instead of the _run method. This allows us to create tools that can perform asynchronous operations, such as making API calls or performing long-running computations, without blocking the main thread of execution. This can be useful for creating tools that need to perform time-consuming tasks or that need to interact with external services. By using async tools, we can improve the performance and responsiveness of our agents, as they can continue to process other tasks while waiting for the async tool to complete its operation.
class Multiply(BaseTool):
    name = "multiply"
    description = "Multiplies two numbers together. Use this tool when you need to multiply two numbers and return the result."
    args_schema: type[BaseModel] = MultiplyArgs

    def _run(self, a: int, b: int) -> int:
        """Multiply two numbers together and return the result."""
        return a * b
    
mutiply_tool = Multiply()

result = mutiply_tool.invoke({"a": 3, "b": 4})
print(result)