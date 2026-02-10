import chromadb

chroma_client = chromadb.Client()

client = chromadb.PersistentClient(path="/Users/lapin/Desktop/RAG Emprez/Chromadb/emprez_docs")
collection = chroma_client.get_or_create_collection(
    name="Documentation_Empreza", 
    metadata={"Description" : "Test Chroma DB pour RAG Emprez"},
    configruation ={
        "hsnw": {
        "ef_contruction": 250,
        "ef_search": 250,
        "max_neighbors": 16,
        "hnsw:space": "cosine",
        "hnsw:ef_construction": 200,
        "hnsw:m": 16
        }
    },
    embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
)

collection.upsert(
    documents=[
)