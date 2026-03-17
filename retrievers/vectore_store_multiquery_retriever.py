# --- Imports ---
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
import os

# --- Load environment ---
# Required:
#   AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
#   AZURE_OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
#   AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<your-embedding-deployment-name>
#   AZURE_OPENAI_CHAT_DEPLOYMENT=<your-chat-deployment-name>  # e.g., gpt-4o-mini
#   (Optional) AZURE_OPENAI_API_VERSION=2024-02-15-preview
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# --- Safety checks ---
missing = [
    name for name, val in [
        ("AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT),
        ("AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY),
        ("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", AZURE_OPENAI_EMBEDDING_DEPLOYMENT),
        ("AZURE_OPENAI_CHAT_DEPLOYMENT", AZURE_OPENAI_CHAT_DEPLOYMENT),
    ] if not val
]
if missing:
    raise ValueError(f"Missing environment var(s): {', '.join(missing)}")

# --- Embeddings ---
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    api_version=AZURE_OPENAI_API_VERSION,
)

# --- Example documents ---
documents = [
    Document(page_content="Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.", metadata={"source": "doc1"}),
    Document(page_content="MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.", metadata={"source": "doc2"}),
    Document(page_content="Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.", metadata={"source": "doc3"}),
    Document(page_content="Rohit Sharma is known for his elegant batting and record-breaking double centuries.", metadata={"source": "doc4"}),
    Document(page_content="Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers.", metadata={"source": "doc5"}),
]

# --- Build FAISS vector store ---
vectorstore = FAISS.from_documents(documents, embeddings)

# --- Base retriever (MMR) ---
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5},  # diversify results
)

# --- LLM for multi-query generation (Azure OpenAI Chat) ---
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.3,      # slight diversity in generations
)

# --- MultiQueryRetriever ---
# It will generate multiple alternative phrasings of your question, run retrieval for each, and merge results.
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
    # Optional: provide a custom prompt to bias how the queries are generated
    # prompt=MultiQueryRetriever.get_default_prompt(),  # default is fine
    # Also optional: set "include_original=True" to always include the original user query
    include_original=True,
    # Optional: control how many queries to generate (default 3)
    # chain_type="stuff"
)

# --- Run a query ---
query = "Who is the best Indian batsman?"

# get_relevant_documents -> returns List[Document]
results = multi_retriever.get_relevant_documents(query)

# --- Print results (deduplicated by page_content for display clarity) ---
seen = set()
print("\n=== Retrieved Documents (MultiQuery + MMR) ===")
for i, doc in enumerate(results, 1):
    key = (doc.page_content, doc.metadata.get("source"))
    if key in seen:
        continue
    seen.add(key)
    print(f"{i}. {doc.page_content}  (source: {doc.metadata.get('source')})")

# --- (Optional) Peek at the sub-queries generated ---
# MultiQueryRetriever exposes the prompt template; to see actual generated queries,
# you can run the inner query-gen chain directly if needed:
def preview_generated_queries(question: str):
    # Internal template typically creates separate lines with queries.
    # We can invoke the underlying LLM with a compatible prompt to preview.
    prompt = (
        "You are a helpful assistant that generates multiple search queries to retrieve "
        "relevant documents. Generate 3 diverse rephrasings for the user question.\n\n"
        f"User question: {question}\n\n"
        "Return each query on a new line without numbering."
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    text = resp.content.strip()
    print("\n--- Generated Sub-Queries (Preview) ---")
    for line in text.splitlines():
        if line.strip():
            print("-", line.strip())

# Uncomment to preview:
# preview_generated_queries(query)