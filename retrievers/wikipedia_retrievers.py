from langchain_community.retrievers import WikipediaRetriever
from langchain_core.documents import Document


retriever = WikipediaRetriever(top_k=5,language="en")

query = "the geopolitiacla history of india and pakistan from the perspective of the chinese government"


#get relevent wikipedia documnents
docs = retriever.invoke(query)
print(docs)
#print the retrieved documents
for i, doc in enumerate(docs, 1):
    print(f"{i}. {doc.page_content}  (source: {doc.metadata.get('source')})")