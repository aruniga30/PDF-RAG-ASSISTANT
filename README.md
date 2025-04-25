# PDF RAG Assistant

A Retrieval-Augmented Generation (RAG) system for querying PDF documents using large language models. This application allows users to upload PDF documents, process them, and ask questions about their content.

## Features

- **PDF Processing**: Extract text from PDF documents while preserving page structure and table of contents information
- **Document Chunking**: Divide documents into manageable chunks for efficient retrieval
- **Vector Search**: Create embeddings for document chunks and perform semantic search
- **LLM Integration**: Query multiple LLM providers including Gemini and SambaNova Cloud
- **Document Viewer**: Built-in PDF viewer with page navigation
- **Document Summarization**: Generate summaries of uploaded documents
- **User Feedback**: Log user feedback on system responses for continual improvement
- **Configurable Settings**: Adjust embedding models, chunk sizes, and retrieval parameters

## Project Structure

- **app.py**: Main Streamlit application with UI components
- **pdf_processor.py**: PDF text extraction and document chunking functionality
- **vector_store.py**: Vector database and embedding model management
- **rag_pipeline.py**: Core RAG functionality including context formatting and LLM querying
- **ui_helpers.py**: Helper functions for UI rendering

## Requirements

- Python 3.8+
- Streamlit
- LangChain
- pdfplumber
- PyMuPDF (fitz)
- HuggingFace Transformers
- OpenAI API client
- Google GenerativeAI

## Installation

1. Clone the repository
2. Install dependencies
3. Fill in required API keys and configuration values in the .env file
4. Run the application using Streamlit

## Usage

1. **Upload Documents**: Use the "Upload & Query" tab to upload PDF documents
2. **Ask Questions**: Enter questions about the document content
3. **View Documents**: Use the "Document Viewer" tab to browse uploaded PDFs
4. **Configure Settings**: Adjust embedding models, chunk sizes, and LLM settings in the "Settings" tab
5. **Generate Summaries**: Create document summaries from the "Upload & Query" tab

## How It Works

1. PDF documents are processed and split into smaller chunks
2. Chunks are embedded using a language model and stored in a vector database
3. When a user asks a question, relevant chunks are retrieved from the vector database
4. Retrieved chunks are formatted into context information for the LLM
5. The LLM generates an answer based on the context and the user's question
6. The answer and source documents are displayed to the user

## Customization

- **Embedding Models**: Choose from various HuggingFace embedding models
- **Vector Stores**: Switch between Chroma (persistent) and FAISS (in-memory)
- **LLM Providers**: Select between Gemini and SambaNova Cloud
- **Chunking Parameters**: Adjust chunk size and overlap for document processing

## Demo
https://drive.google.com/file/d/1hMpRDL62t9Lj34IH4zWZqdUb8PMKkeZX/view?usp=sharing
