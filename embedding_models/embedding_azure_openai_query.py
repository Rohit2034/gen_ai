from langchain_openai import AzureOpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="gpt-oss-120b",  # Replace with your actual embedding deployment name
    api_version="2024-02-01",
)

result = embeddings.embed_query("What is the capital of France?")
print(result)
