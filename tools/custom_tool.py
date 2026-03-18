from langchain_core.tools import tool


@tool
def multiply(a: float, b: float) ->  float:
    # highly recommended to add doc strings to your tools, as they are used in the agent's reasoning process to determine which tool to use
    """Multiply two numbers"""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


# result = multiply.invoke(3, 4)

# print("Multiply result:", result)
#always pass in dictionary when invoking a tool, even if it only has one parameter. This is because the tool's signature is defined in terms of keyword arguments, and the agent will invoke it with a dictionary of arguments.
result = add.invoke({"a":3, "b":4})
print(multiply.name)
print(multiply.description)
print(multiply.args)

#llm agents will use the tool's name, description, and args to determine which tool to use when responding to a user query. The name is used to identify the tool, the description is used to understand what the tool does, and the args are used to understand what parameters the tool expects.
#so we send this schema to the agent, and it uses it to determine how to call the tool when it needs to use it. The schema includes the name of the tool, a description of what it does, and the arguments it expects. This allows the agent to use the tool effectively when responding to user queries.
print(multiply.args_schema.model_json_schema())

print("Add result:", result)