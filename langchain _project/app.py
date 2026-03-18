from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()  # Ensure your .env has AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, OPENAI_API_VERSION

# ----- Configuration -----
VIDEO_ID = "dQw4w9WgXcQ"  # Replace with your YouTube video ID
# Replace these with your actual Azure deployment names from Azure OpenAI Studio
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")            # e.g., gpt-4o
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")

API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-02-01")

# ----- Fetch transcript -----
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(VIDEO_ID, languages=["en"])
    transcript = " ".join([seg["text"] for seg in transcript_list])
    print("Transcript loaded. Length:", len(transcript))
except Exception as e:
    print(f"Transcript not found or error occurred: {e}")
    transcript = ""  # Avoid NameError later

if not transcript.strip():
    raise RuntimeError("No transcript available to build the vector store.")

# ----- Split into chunks -----
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(transcript)
print("Chunks:", len(chunks))

# ----- Embeddings & Vector store -----
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=EMBED_DEPLOYMENT,
    api_version=API_VERSION,
)

vectorstore = FAISS.from_texts(chunks, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ----- LLM -----
llm = AzureChatOpenAI(
    azure_deployment=CHAT_DEPLOYMENT,
    api_version=API_VERSION,
    temperature=0.7,
)

# ----- Prompt -----
prompt_template = PromptTemplate(
    template=(
        "You are a helpful assistant that answers questions based on the retrieved context "
        "from a YouTube video transcript. If you don't know the answer, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    ),
    input_variables=["context", "question"],
)

# ----- Simple RAG call -----
question = "Is the topic of aliens discussed in this video? If yes, what was discussed?"
retrieved_docs = retriever.invoke(question)
context = "\n\n".join(doc.page_content for doc in retrieved_docs)
final_prompt = prompt_template.format(context=context, question=question)
response = llm.invoke(final_prompt)
print("\n--- LLM Response (simple) ---\n", response.content)

# ----- Chains (Runnable graph) -----
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

# Test the parallel chain
_ = parallel_chain.invoke("Who is Demis Hassabis?")  # This returns a dict with context & question

parser = StrOutputParser()
main_chain = parallel_chain | prompt_template | llm | parser

print("\n--- LLM Response (main_chain) ---\n", main_chain.invoke("What is DeepMind?"))