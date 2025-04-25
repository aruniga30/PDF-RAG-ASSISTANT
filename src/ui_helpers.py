import base64
import fitz
from io import BytesIO
import streamlit as st

def display_pdf_page(file_path, page_num):
    """Render a PDF page as HTML for display"""
    try:
        with fitz.open(file_path) as pdf:
            page = pdf[page_num-1]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_base64 = base64.b64encode(pix.tobytes("png")).decode()
            
        return f"""
        <div style="border:1px solid #ddd; padding:3px; border-radius:3px;">
            <img src="data:image/png;base64,{img_base64}" style="width:100%">
        </div>
        """
    except Exception as e:
        return f"<div>Error displaying PDF page: {str(e)}</div>"
    

def get_page_image(pdf_path, page_num):
    """Extract an image of the specified page from a PDF"""
    try:
        with fitz.open(pdf_path) as doc:
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img_b64 = base64.b64encode(pix.tobytes("png")).decode()
            return f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%;"/>'
    except Exception as e:
        return f"Error rendering page image: {str(e)}"
