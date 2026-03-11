from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

from dotenv import load_dotenv


load_dotenv()

model = AzureChatOpenAI(
        azure_deployment="gpt-oss-120b",    
        model="gpt-4",
        api_version="2024-02-15-preview",       
        temperature=0,
    )
chat_history = [
    SystemMessage(content="You are a helpful assistant."),

]




while True:
    user_input = input("User: ")
    # brute force approach
    chat_history.append(HumanMessage(content=user_input))
    if user_input=='exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ", result.content)

print("Chat ended.",chat_history)