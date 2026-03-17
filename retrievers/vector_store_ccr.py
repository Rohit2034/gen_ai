# --- Imports ---
from dotenv import load_dotenv
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

# Text splitting for long documents
# If your langchain version uses a different import:
#   from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Contextual compression pieces
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

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

# --- Initialize models ---
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    api_version=AZURE_OPENAI_API_VERSION,
)

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.2,   # lower = more conservative extraction
)

# --- New LONG documents ---
# These are intentionally lengthy so you can see compression in action.
# In real scenarios, use your actual content.
kohli_long = (
    "Virat Kohli is an Indian cricketer renowned for his prolific batting across formats. "
    "He has captained India in Tests, ODIs, and T20Is, leading with intensity and fitness-first ethos. "
    "Over the years, Kohli amassed runs in pressure chases, often pacing innings expertly. "
    "In ODI cricket, his knack for run-chases earned comparisons with the greats, while in Tests he "
    "translated form into away tours in Australia and England. "
    "In the IPL, Kohli has delivered standout seasons, including a record-breaking run tally. "
    "Analysts often highlight his cover drives, quick singles, and fitness, which contribute to high consistency. "
    "Beyond pure stats, his competitive edge and standards have shaped the team’s culture. "
    "Debates about the 'best Indian batsman' often include Sachin Tendulkar for his longevity and records, "
    "Virat Kohli for modern-era dominance, and Rohit Sharma for limited-overs explosiveness. "
    "Kohli’s centuries in key matches and ICC tournaments are frequently cited as evidence of clutch performance. "
    "He has also adapted his game from aggressive stroke-play to calculated risk-management when needed. "
) * 3  # repeat to make it long

sachin_long = (
    "Sachin Tendulkar, widely regarded as one of the greatest batters of all time, "
    "held numerous batting records including the most international runs and centuries. "
    "Spanning over two decades, his career bridged generations of cricket fans. "
    "He excelled in all conditions: from seaming tracks in England to bouncy pitches in Australia. "
    "Known for technical purity—balance, straight bat, late cuts—Tendulkar’s mastery was evident against pace and spin. "
    "His ODI double century and the 2011 World Cup win were iconic milestones toward the twilight of his career. "
    "Conversations on the 'best Indian batsman' frequently revolve around his longevity and unmatched statistical mountain. "
    "Even post-retirement, his legacy informs how new talents are measured. "
) * 3

rohit_long = (
    "Rohit Sharma, India’s all-format captain, is celebrated for his elegance and timing. "
    "He owns multiple double centuries in ODIs—an unprecedented feat—showcasing his ability to convert starts into massive scores. "
    "In T20Is, he has excelled both as an opener and a middle-order batter, adapting to team needs. "
    "Rohit’s pull shot against pace is among the most aesthetically pleasing strokes in modern cricket. "
    "As a leader, he emphasizes clarity of roles and aggressive but measured tactics. "
    "Debates around 'best Indian batsman' often cite his dominance in white-ball cricket and his red-ball resurgence as an opener. "
) * 3

dhoni_long = (
    "MS Dhoni, though primarily renowned for finishing and leadership, contributed crucial runs across formats. "
    "His calm demeanor under pressure made him one of the finest finishers in limited-overs history. "
    "While not always at the top of run charts, his impact on outcomes, field placements, and team balance is immense. "
    "Discussions around 'best batsman' still acknowledge Dhoni’s unique influence and match awareness. "
) * 3

bumrah_long = (
    "Jasprit Bumrah, primarily a fast bowler, revolutionized death bowling with yorkers and deceptive pace. "
    "Though not central to batting debates, his lower-order runs sometimes swing momentum. "
) * 3

documents = [
    Document(page_content=kohli_long, metadata={"source": "kohli_profile"}),
    Document(page_content=sachin_long, metadata={"source": "sachin_profile"}),
    Document(page_content=rohit_long, metadata={"source": "rohit_profile"}),
    Document(page_content=dhoni_long, metadata={"source": "dhoni_profile"}),
    Document(page_content=bumrah_long, metadata={"source": "bumrah_profile"}),
]

# --- Chunk long docs before indexing ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,     # tune per token budget
    chunk_overlap=120,  # overlap helps preserve context across chunks
)
splits = text_splitter.split_documents(documents)

# --- Build FAISS from chunks ---
vectorstore = FAISS.from_documents(splits, embeddings)

# --- Base retriever (MMR for diversity) ---
base_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "lambda_mult": 0.5},
)

# --- OPTION A: LLM-based contextual compression (simplest) ---
llm_extractor = LLMChainExtractor.from_llm(llm)

cc_retriever_llm = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=llm_extractor,
)

# --- OPTION B (optional): Embeddings filter + LLM extractor pipeline ---
# First filter by semantic similarity, then extract relevant spans using LLM.
emb_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.72,  # tune 0.65–0.8 based on recall/precision needs
)
compressor_pipeline = DocumentCompressorPipeline(
    transformers=[emb_filter, llm_extractor]
)

cc_retriever_pipeline = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor_pipeline,
)

# --- Run queries ---
query = "Who is the best Indian batsman and why?"

print("\n=== Base Retrieval (pre-compression) ===")
base_docs = base_retriever.get_relevant_documents(query)
for i, d in enumerate(base_docs, 1):
    print(f"{i}. [{d.metadata.get('source')}] chars={len(d.page_content)}\n   {d.page_content[:220]}...\n")

print("\n=== Contextual Compression (LLM extractor) ===")
compressed_docs_a = cc_retriever_llm.get_relevant_documents(query)
for i, d in enumerate(compressed_docs_a, 1):
    print(f"{i}. [{d.metadata.get('source')}] chars={len(d.page_content)}\n   {d.page_content[:220]}...\n")

print("\n=== Contextual Compression (Embeddings filter + LLM extractor) ===")
compressed_docs_b = cc_retriever_pipeline.get_relevant_documents(query)
for i, d in enumerate(compressed_docs_b, 1):
    print(f"{i}. [{d.metadata.get('source')}] chars={len(d.page_content)}\n   {d.page_content[:220]}...\n")