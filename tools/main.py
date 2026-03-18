from langchain_community.tools import DuckDuckGoSearchRun

# If this raises ImportError, the dependency wasn't found in THIS interpreter
tool = DuckDuckGoSearchRun()

print("Tool created OK. Running a query...")
print(tool.invoke("What is the capital of France?"))


from langchain_community.tools import ShellTool
shell_tool = ShellTool()
results = shell_tool.invoke("echo Hello, World!")
print(results)