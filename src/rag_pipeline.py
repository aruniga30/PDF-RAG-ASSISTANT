import csv
import os
from typing import List, Dict
from langchain.schema import Document
from openai import OpenAI
import google.generativeai as genai

from src.vector_store import get_retriever

def format_context_from_docs(docs: List[Document]) -> str:
    """Format retrieved documents into context string with metadata"""
    context_parts = []
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        source = doc.metadata.get("source_file", "Unknown")
        section = doc.metadata.get("section", "")
        
        section_info = f" - Section: {section}" if section else ""
        header = f"[Document: {source}, Page: {page}{section_info}]"
        
        context_parts.append(f"{header}\n{doc.page_content}\n")
    
    return "\n\n".join(context_parts)

def query_llm(prompt: str, provider: str, api_key: str, model: str) -> str:
    """Query appropriate LLM based on provider selection"""
    try:
        if provider == "Gemini":
            genai.configure(api_key=api_key)
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(prompt)
            return response.text
            
    
        elif provider == "SambaNova Cloud":
            client = OpenAI(
                api_key=api_key, 
                base_url=os.getenv("SAMBANOVA_BASE_URL")
            )
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that provides accurate information from documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content
            
        return "Unknown provider selected"
    except Exception as e:
        return f"Error querying {provider}: {str(e)}"

def generate_answer(query: str, vector_store, top_k: int, provider: str, api_key: str, model: str) -> Dict:
    """Generate an answer using RAG approach"""
    if not vector_store:
        return {"result": "No documents loaded. Please upload a PDF first.", "source_documents": []}

    # Retrieve relevant documents
    retriever = get_retriever(vector_store, k=top_k)
    retrieved_docs = retriever.invoke(query)
    context = format_context_from_docs(retrieved_docs)

    # LLM prompt template
    template = """
    Answer the question based only on the following context. If the answer cannot be found in the context, state that you don't have enough information to answer accurately.

    Context:
    {context}

    Question: {question}

    Instructions:
    - Answer the question concisely and accurately based on the provided context
    - Include source references (document name, page number) to support your answer
    - Format source citations like this: [Document: page X]
    - Do not make up information that isn't present in the context

    Answer:
    """

    prompt = template.format(context=context, question=query)
    response = query_llm(prompt, provider, api_key, model)

    return {
        "result": response,
        "source_documents": retrieved_docs
    }

def generate_document_summary(processed_docs: List[Document], provider: str, model: str, 
                              api_key: str = None, doc_name: str = None) -> str:
    """Generate summary of documents"""
    # Filter documents if doc_name is provided
    docs_to_summarize = [doc for doc in processed_docs if 
                        not doc_name or doc.metadata.get("source_file") == doc_name]
    
    if not docs_to_summarize:
        return "No documents available to summarize."

    # Create a summary prompt with document content
    all_text = ""
    for doc in docs_to_summarize:
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        all_text += f"--- Document: {source}, Page: {page} ---\n{doc.page_content}\n\n"

    # Limit text size for LLM context window
    max_chars = int(os.getenv("MAX_SUMMARY_CHAR"))
    if len(all_text) > max_chars:
        all_text = all_text[:max_chars] + "...[text truncated due to length]"

    summary_prompt = f"""
    Please provide a comprehensive summary of the following document content:

    {all_text}

    Instructions:
    - Identify and highlight the main topics and key points
    - Structure the summary with clear sections
    - Include important facts, figures, and conclusions
    - Keep the summary concise but informative

    Summary:
    """

    return query_llm(summary_prompt, provider, api_key, model)

def init_feedback_log():
    """Initialize feedback log file if it doesn't exist"""
    os.makedirs("logs", exist_ok=True)

    feedback_file = os.getenv("LOG_FILE_NAME")

    if not os.path.exists(feedback_file):
        with open(feedback_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["query", "response", "rating", "document", "retrieved_chunks"])

def log_user_feedback(query: str, response: str, rating: str, doc_name: str = "", chunks: List[Document] = None):
    """Log user feedback to CSV file"""
    chunks_str = ""
    if chunks:
        chunks_str = "; ".join([f"{doc.metadata.get('source_file')}:p{doc.metadata.get('page')}" 
                              for doc in chunks])
        
    feedback_file = os.getenv("LOG_FILE_NAME")

    with open(feedback_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([query, response, rating, doc_name, chunks_str])

def get_document_names(processed_docs: List[Document]) -> List[str]:
    """Get list of loaded document names"""
    return list({doc.metadata.get("source_file", "") for doc in processed_docs})