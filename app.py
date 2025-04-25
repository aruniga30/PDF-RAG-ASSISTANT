import os
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
from io import BytesIO

# Load custom modules
from src.pdf_processor import process_pdf
from src.vector_store import create_vector_store, add_documents
from src.rag_pipeline import (
    generate_answer, generate_document_summary,
    init_feedback_log, log_user_feedback, get_document_names
)
from src.ui_helpers import display_pdf_page, get_page_image

# Load environment variables
load_dotenv()


def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "embedding_model": os.getenv("DEFAULT_EMBEDDING_MODEL"),
        "chunk_size": int(os.getenv("DEFAULT_CHUNK_SIZE")),
        "chunk_overlap": int(os.getenv("DEFAULT_CHUNK_OVERLAP")),
        "top_k": int(os.getenv("DEFAULT_TOP_K")),
        "llm_provider": os.getenv("DEFAULT_LLM_PROVIDER"),
        "llm_model": os.getenv("DEFAULT_LLM_MODEL"),
        "vector_store": None,
        "processed_docs": [],
        "chunked_docs": [],
        "current_pdf_path": None,
        "uploaded_files": [],
        "documents_loaded": False
    }
    
    # Set API key based on provider if available
    if "api_key" not in st.session_state and "llm_provider" in st.session_state:
        provider_key = f"{st.session_state.llm_provider.replace(" ", "_").upper()}_API_KEY"
        defaults["api_key"] = os.getenv(provider_key, "")
    
    # Initialize all missing session state variables
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files"""
    with st.spinner("Processing documents..."):
        total_chunks = 0
        for uploaded_file in uploaded_files:
            # Skip if already processed
            if any(f["name"] == uploaded_file.name for f in st.session_state.uploaded_files):
                continue

            # Save file to temp directory
            temp_dir = "temp_dir"
            os.makedirs(temp_dir, exist_ok=True)
            tmp_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())

            # Process document
            docs, chunks = process_pdf(
                tmp_path, 
                st.session_state.chunk_size, 
                st.session_state.chunk_overlap
            )

            # Update session state
            st.session_state.processed_docs.extend(docs)
            st.session_state.chunked_docs.extend(chunks)
            total_chunks += len(chunks)

            # Store file info
            st.session_state.uploaded_files.append({
                "name": uploaded_file.name,
                "path": tmp_path
            })

            # Set current PDF path if not set
            if not st.session_state.current_pdf_path:
                st.session_state.current_pdf_path = tmp_path

        # Create or update vector store
        if st.session_state.vector_store is None:
            st.session_state.vector_store = create_vector_store(
                st.session_state.chunked_docs,
                st.session_state.embedding_model,
            )
        else:
            st.session_state.vector_store = add_documents(
                st.session_state.vector_store,
                st.session_state.chunked_docs,
                st.session_state.embedding_model
            )

        st.session_state.documents_loaded = True
        return total_chunks

def display_answer_with_sources(user_query, result):
    """Display answer and source documents"""
    st.markdown("### Answer")
    st.write(result["result"])

    st.markdown("### Sources")
    for i, doc in enumerate(result["source_documents"]):
        source_file = doc.metadata.get("source_file", "Unknown")
        page_num = doc.metadata.get("page", "Unknown")
        section = doc.metadata.get("section", "")

        with st.expander(f"Source {i+1}: {source_file} (Page {page_num})"):
            if section:
                st.caption(f"Section: {section}")
            
            # Find source file path
            source_path = next((
                f["path"] for f in st.session_state.uploaded_files
                if os.path.basename(f["path"]) == os.path.basename(source_file)
            ), None)
            
            # Display page image if available
            if source_path and isinstance(page_num, int):
                page_img_html = get_page_image(source_path, page_num - 1)
                st.markdown(page_img_html, unsafe_allow_html=True)
            else:
                st.text(doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""))
                st.caption("(Image preview not available)")

            # Feedback system
            col_fb1, col_fb2 = st.columns([3, 1])
            with col_fb1:
                rating = st.select_slider(
                    f"Rate relevance of source {i+1}:",
                    options=["Not relevant", "Somewhat relevant", "Very relevant"],
                    key=f"rating_{i}"
                )
            with col_fb2:
                if st.button("Submit Feedback", key=f"feedback_{i}"):
                    log_user_feedback(
                        query=user_query,
                        response=result["result"],
                        rating=rating,
                        doc_name=source_file,
                        chunks=result["source_documents"],
                    )
                    st.success("Feedback recorded!")

def render_upload_query_tab():
    """Render the Upload & Query tab"""
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Upload PDF Documents")

        uploaded_files = st.file_uploader(
            "Upload one or more PDF documents",
            type="pdf",
            accept_multiple_files=True
        )

        if uploaded_files and st.button("Process Documents"):
            total_chunks = process_uploaded_files(uploaded_files)
            st.success(f"Processed {len(uploaded_files)} documents with {total_chunks} total chunks.")

        st.subheader("Ask Questions")
        user_query = st.text_input("Enter your question about the documents:")

        submit_query = st.button("Submit Question")
        
        if submit_query and user_query and st.session_state.documents_loaded:
            with st.spinner("Generating answer..."):
                result = generate_answer(
                    user_query,
                    st.session_state.vector_store,
                    st.session_state.top_k,
                    st.session_state.llm_provider,
                    st.session_state.api_key,
                    st.session_state.llm_model,
                )
            display_answer_with_sources(user_query, result)
        elif user_query:
            st.warning("Please click submit to view response")

    with col2:
        if st.session_state.documents_loaded:
            st.subheader("Document Summary")

            doc_names = get_document_names(st.session_state.processed_docs)
            selected_doc = st.selectbox(
                "Select document to summarize (optional)",
                ["All Documents"] + doc_names
            )

            if st.button("Generate Summary"):
                with st.spinner("Generating document summary..."):
                    doc_name = None if selected_doc == "All Documents" else selected_doc

                    summary = generate_document_summary(
                        st.session_state.processed_docs,
                        st.session_state.llm_provider,
                        st.session_state.llm_model,
                        st.session_state.api_key,
                        doc_name
                    )

                st.markdown(summary)
                
                # Download summary option
                summary_bytes = summary.encode()
                st.download_button(
                    label="Download Summary",
                    data=BytesIO(summary_bytes),
                    file_name="document_summary.txt",
                    mime="text/plain"
                )

def render_document_viewer_tab():
    """Render the Document Viewer tab"""
    st.subheader("PDF Document Viewer")

    if st.session_state.uploaded_files:
        # Document and page selector
        col_doc, col_page = st.columns(2)

        with col_doc:
            file_options = [f["name"] for f in st.session_state.uploaded_files]
            selected_file_name = st.selectbox("Select document", file_options)

            # Find the file path for the selected file
            selected_file = next((
                f for f in st.session_state.uploaded_files 
                if f["name"] == selected_file_name
            ), None)
            
            if selected_file:
                st.session_state.current_pdf_path = selected_file["path"]

        if st.session_state.current_pdf_path:
            # Get page count and display page selector
            with fitz.open(st.session_state.current_pdf_path) as doc:
                page_count = len(doc)

            with col_page:
                selected_page = st.slider("Select page", 1, page_count, 1)

            # Display the selected page
            pdf_html = display_pdf_page(st.session_state.current_pdf_path, selected_page)
            st.markdown(pdf_html, unsafe_allow_html=True)

            # Display text for the current page
            with fitz.open(st.session_state.current_pdf_path) as doc:
                text = doc[selected_page-1].get_text()

            with st.expander("Page Text"):
                st.text(text)
    else:
        st.info("Upload PDF documents to view them here.")

def render_settings_tab():
    """Render the Settings tab"""
    st.subheader("Pipeline Settings")

    col_embed, col_llm = st.columns(2)

    # Store original values to compare for changes
    original_embedding_model = st.session_state.embedding_model
    original_chunk_size = st.session_state.chunk_size
    original_chunk_overlap = st.session_state.chunk_overlap

    with col_embed:
        st.write("#### Embedding Settings")

        embedding_model = st.selectbox(
            "Embedding Model",
            [
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/multi-qa-mpnet-base-dot-v1",
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "sentence-transformers/all-distilroberta-v1"
            ],
            index=0
        )

        chunk_size = st.slider("Chunk Size", 100, 1000, st.session_state.chunk_size, 50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 200, st.session_state.chunk_overlap, 10)

    with col_llm:
        st.write("#### LLM Settings")
        llm_type = st.selectbox(
            "LLM Provider",
            ["SambaNova Cloud", "Gemini"]
        )

        # Different API key inputs based on provider
        if llm_type == "Gemini":
            default_key = os.getenv("GEMINI_API_KEY", "")
            api_key = st.text_input("Gemini API Key", value=default_key, type="password")
            model_options = ["gemini-1.5-pro", "gemini-2.5-pro-preview-03-25", "gemini-1.5-flash", "gemini-2.0-flash"]
        else:  # SambaNova
            default_key = os.getenv("SAMBANOVA_CLOUD_API_KEY", "")
            api_key = st.text_input("SambaNova API Key", value=default_key, type="password")
            model_options = [
                "Meta-Llama-3.2-3B-Instruct", "DeepSeek-V3-0324", "Llama-3.3-Swallow-70B-Instruct-v0.4", 
                "Llama-4-Scout-17B-16E-Instruct", "Meta-Llama-3.1-405B-Instruct",
                "Meta-Llama-3.1-8B-Instruct", "Meta-Llama-3.2-1B-Instruct",
                 "Meta-Llama-3.3-70B-Instruct", 
                "Meta-Llama-Guard-3-8B"
            ]

        model = st.selectbox("Model", model_options)
        retrieval_k = st.slider("Number of chunks to retrieve", 1, 10, st.session_state.top_k)


    if st.button("Update Settings"):
        # Check if embedding-related settings have changed
        embedding_settings_changed = (
            embedding_model != original_embedding_model or
            chunk_size != original_chunk_size or
            chunk_overlap != original_chunk_overlap
        )
        
        # Save settings to session state
        st.session_state.embedding_model = embedding_model
        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap
        st.session_state.top_k = retrieval_k
        st.session_state.llm_provider = llm_type
        st.session_state.llm_model = model
        st.session_state.api_key = api_key

        # Only reprocess documents if embedding-related settings have changed
        if embedding_settings_changed and st.session_state.uploaded_files and st.session_state.processed_docs:
            with st.spinner("Reprocessing documents with new embedding settings..."):
                # Reset state
                st.session_state.processed_docs = []
                st.session_state.chunked_docs = []
                st.session_state.vector_store = None

                # Reprocess each file
                for file_info in st.session_state.uploaded_files:
                    docs, chunks = process_pdf(
                        file_info["path"], 
                        chunk_size, 
                        chunk_overlap
                    )
                    st.session_state.processed_docs.extend(docs)
                    st.session_state.chunked_docs.extend(chunks)

                # Recreate vector store
                st.session_state.vector_store = create_vector_store(
                    st.session_state.chunked_docs,
                    embedding_model,
                )
            st.success(f"Embedding settings changed - documents reprocessed with {len(chunks)} total chuncks!" )
        else:
            st.success("Settings updated successfully!")

def main():
    st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
    
    # Initialize session state
    init_session_state()
    
    st.title("📄 PDF Question-Answering System")
    
    # Create tabs for different features
    tab1, tab2, tab3 = st.tabs(["Upload & Query", "Document Viewer", "Settings"])
    
    # Initialize feedback log
    init_feedback_log()
    
    # Render tabs
    with tab1:
        render_upload_query_tab()
    
    with tab2:
        render_document_viewer_tab()
    
    with tab3:
        render_settings_tab()

if __name__ == "__main__":
    main()