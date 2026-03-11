from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.documents import Document
import os

from langchain_openai import AzureOpenAIEmbeddings
load_dotenv()
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),     
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    api_version="2024-02-15-preview",
)



documents = [
    Document(page_content="Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.", metadata={"source": "doc1"}),
    Document(page_content="MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.", metadata={"source": "doc2"}),   
    Document(page_content="Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.", metadata={"source": "doc3"}),
    Document(page_content="Rohit Sharma is known for his elegant batting and record-breaking double centuries.", metadata={"source": "doc4"}),
    Document(page_content="Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers.", metadata={"source": "doc5"}),
]
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k": 3, "lambda_mult": 0.5}
)

query = "Who is the best Indian batsman?"
results = retriever.invoke(query)

for i ,doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}  (source: {doc.metadata.get('source')})")   
    