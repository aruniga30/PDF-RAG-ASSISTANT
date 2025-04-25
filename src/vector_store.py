import os
from typing import List, Any, Optional
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma, FAISS

def get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Create and return an embedding model instance"""
    return HuggingFaceEmbeddings(model_name=model_name)

def create_vector_store(docs: List[Document], embedding_model_name: str) -> Optional[Any]:
    """Create and return a vector store from documents"""
    if not docs:
        return None
        
    embedding_model = get_embedding_model(embedding_model_name)  
    store_type = os.getenv("VECTOR_STORE_TYPE")

    if store_type.lower() == "chroma":
        return Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory= os.getenv("PERSIST_DIRECTORY")
        )
    elif store_type.lower() == "faiss":
        return FAISS.from_documents(
            documents=docs,
            embedding=embedding_model
        )
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")

def get_retriever(vector_store, k: int):
    """Get a retriever interface for the vector store"""
    return vector_store.as_retriever(search_kwargs={"k": k}) if vector_store else None

def add_documents(vector_store, docs: List[Document], embedding_model_name: str):
    """Add new documents to existing vector store"""
    if not vector_store or not docs:
        return vector_store
        
    if hasattr(vector_store, "add_documents"):
        vector_store.add_documents(docs)
        return vector_store
    else:
        # For FAISS which doesn't support add_documents directly
        embedding_model = get_embedding_model(embedding_model_name)
        current_docs = vector_store.docstore.get_all_documents()
        return FAISS.from_documents(current_docs + docs, embedding_model)