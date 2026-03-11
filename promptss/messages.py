from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

from langchain_openai import AzureChatOpenAI

from dotenv import load_dotenv

load_dotenv()

model = AzureChatOpenAI(
        azure_deployment="gpt-oss-120b",    
        model="gpt-4",
        api_version="2024-02-15-preview",
        temperature=0,
    )

messagess = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of India?"),
    
]


result = model.invoke(messagess)
messagess.append(AIMessage(content=result.content))

print(messagess)
