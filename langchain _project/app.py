from youtube_transcript_api import YouTubeTranscriptApi
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

try:
        
    video_id = "dQw4w9WgXcQ"  # Replace with your YouTube video ID
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id,languages=['en'])
    # Combine transcript segments into a single string
    transcript = " ".join([segment['text'] for segment in transcript_list])
    print(transcript)
except YouTubeTranscriptApi.TranscriptNotFound:
    print("Transcript not found for the given video ID.")



splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(transcript)
len(chunks)

# Initialize the Azure OpenAI embeddings model


embeddings = AzureOpenAIEmbeddings(
    azure_deployment="gpt-oss-120b",  # Replace with your actual embedding deployment name
    api_version="2024-02-01",
)
vectorstore = FAISS.from_texts(chunks, embeddings)





retriever = vectorstore.as_retriever(search_type ="similarity", search_kwargs={"k": 3})
retriever.invoke("what is deepmind")


llm = AzureChatOpenAI(
    azure_deployment="gpt-oss-120b",  # Replace with your actual
    api_version="2024-02-01",
    temperature=0.7,
)
prompt_template = PromptTemplate(
    template = """
you are a helpful assistant that answers questions based on the following retrieved context from a youtube video transcript. If you don't know the answer, say you don't know.

Context: {context}
Question: {question}
""",
input_variables=["context","question"]     

)


question = "is the topic of aliens discussed in this video? if yes then what was disscussed about it?"
retrieved_docs = retriever.invoke(question)
context = "\n\n".join(doc.page_content for doc in retrieved_docs)
final_prompt = prompt_template.format(context=context, question=question)
response = llm.invoke(final_prompt)


print(response.content)


################################################
# using chains

def format_docs(retrieved_docs):
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context


parrale_chain = RunnableParallel({
    context: retriever | RunnableLambda(format_docs),
    question: RunnablePassthrough()
})

parrale_chain.invoke("who is Demis Hassabis?")


parser = StrOutputParser()

main_chain = parrale_chain|prompt_template | llm | parser

print(main_chain.invoke("what is deepmind?"))