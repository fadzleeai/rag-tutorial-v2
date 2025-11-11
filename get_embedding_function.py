
# from langchain_community.embeddings.ollama import OllamaEmbeddings 
# from langchain_community.embeddings.bedrock import BedrockEmbeddings

# Use the new import for HuggingFaceEmbeddings from the langchain-huggingface package
from langchain_huggingface import HuggingFaceEmbeddings 

def get_embedding_function():
    # We use a standard, lightweight sentence-transformers model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Initialize the embeddings using HuggingFaceEmbeddings
    # This runs the model in-process within your Python script
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    return embeddings

if __name__ == "__main__":
    emb_func = get_embedding_function()
    print("HuggingFace Embedding function initialized successfully!")
    
    # Test an embedding
    query = "How to use sentence transformers?"
    query_result = emb_func.embed_query(query)
    print(f"Embedding length: {len(query_result)}")
    print(f"First 5 dimensions: {query_result[:5]}")