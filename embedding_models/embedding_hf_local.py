from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name=r'C:\Users\303370\.cache\chroma\onnx_models\all-MiniLM-L6-v2\onnx', model_kwargs={'backend': 'onnx'})

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

vector = embedding.embed_documents(documents)

print(str(vector))