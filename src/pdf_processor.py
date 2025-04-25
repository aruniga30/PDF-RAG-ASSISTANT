import os
import uuid
from typing import List, Dict
import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_pdf(file_path: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Process PDF file and return combined document chunks"""
    processed_doc = extract_text_from_pdf(file_path)
    if chunk_size > 0:
        docs = split_documents(processed_doc, chunk_size, chunk_overlap)
    return processed_doc, docs

def extract_text_from_pdf(file_path: str) -> List[Document]:
    """Extract text while preserving page structure using pdfplumber"""
    docs = []
    filename = os.path.basename(file_path)
    
    with pdfplumber.open(file_path) as pdf:
        toc = extract_toc(pdf)
        
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            text = page.extract_text() or ""
            
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "page": page_num,
                        "source_file": filename,
                        "section": get_section_for_page(toc, page_num)
                    }
                ))
    return docs

def extract_toc(pdf) -> List[Dict]:
    """Extract table of contents if available"""
    try:
        toc = []
        if hasattr(pdf, 'outline') and pdf.outline:
            for item in pdf.outline:
                if isinstance(item, dict) and 'dest' in item:
                    page = pdf.get_page_number(item['dest'])
                    toc.append({"title": item.get('title', ''), "page": page + 1})
        return toc
    except:
        return []

def get_section_for_page(toc: List[Dict], page_num: int) -> str:
    """Determine section title for a given page based on TOC"""
    if not toc:
        return ""
    
    current_section = ""
    for item in toc:
        if item['page'] <= page_num:
            current_section = item['title']
        else:
            break
            
    return current_section

def split_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Split documents into smaller chunks while preserving metadata"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunked_docs = []
    for doc in docs:
        for i, split in enumerate(text_splitter.split_text(doc.page_content)):
            metadata = doc.metadata.copy()
            metadata.update({"chunk_id": str(uuid.uuid4()), "chunk_index": i})
            chunked_docs.append(Document(page_content=split, metadata=metadata))

    return chunked_docs