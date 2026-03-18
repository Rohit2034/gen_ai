from langchain_community.tools import tool

#custom tools
@tool
def add(a:int , b:int)->int:
    """Add two numbers together and return the result."""
    return a + b

@tool
def multiply(a:int, b:int)-> int:
    """multiply two numbers"""
    return a*b


class MathToolKit:
    def get_tools(self):
        return [add,multiply]
    

toolkit = MathToolKit()
tools = toolkit.get_tools()

for tool in tools:
    print(tool.name,"   ", tool.description)