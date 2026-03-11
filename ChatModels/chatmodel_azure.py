from langchain_openai import AzureChatOpenAI

from dotenv import load_dotenv

load_dotenv()

model = AzureChatOpenAI(
    azure_deployment="gpt-oss-120b",  
    model="gpt-4",
    api_version="2024-02-15-preview", 
    temperature=0,
)

result = model.invoke("What is the capital of spain?")
print(result.content)


