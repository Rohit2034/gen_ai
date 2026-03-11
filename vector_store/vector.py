from dotenv import load_dotenv
import os

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    api_version="2024-02-15-preview",
)

docs = [
    Document(page_content="Machine learning is a subset of AI...", metadata={"source": "ml_intro_2"}),
    Document(page_content="Natural language processing allows computers to understand...", metadata={"source": "nlp_3"}),
    Document(page_content="Deep learning uses neural networks with multiple layers...", metadata={"source": "dl_5"}),
]

vectorstore = Chroma.from_documents(docs, embeddings)

query = "What is machine learning?"
results = vectorstore.similarity_search(query, k=3)

for i, r in enumerate(results, 1):
    print(f"{i}. {r.page_content}  (source: {r.metadata.get('source')})")