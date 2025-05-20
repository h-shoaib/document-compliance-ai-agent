import streamlit as st
import os
import mimetypes
import traceback 

import pdfplumber
from PIL import Image

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

#these are the functions for text extraction.
#i have used pdfplumber and azure document intelligence api for text extraction

def get_azure_di_client_original():
    azure_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    azure_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    if not azure_endpoint or not azure_key:
        raise ValueError(
            "Azure endpoint and key missing. Ensure AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and "
            "AZURE_DOCUMENT_INTELLIGENCE_KEY are set in your environment or .env file."
        )
    return DocumentIntelligenceClient(endpoint=azure_endpoint, credential=AzureKeyCredential(azure_key))

@st.cache_resource
def get_azure_di_client():
    try:
        client = get_azure_di_client_original()
        
        return client
    except ValueError as ve:
        st.error(f"Configuration Error for Azure DI Client: {ve}")
        return None
    except Exception as e:
        st.error(f"Error initializing Azure DI Client: {e}")
        return None

def ocr_with_azure_document_intelligence_from_bytes(client: DocumentIntelligenceClient, image_bytes: bytes) -> str:
    if not client:
        print("Azure DI client not available for OCR from bytes.")
        return ""
    try:
        print("Attempting Azure OCR from bytes...")
        poller = client.begin_analyze_document(
            "prebuilt-read",
            image_bytes,
            content_type="application/octet-stream"
        )
        result = poller.result()
        print("Azure OCR from bytes successful.")
        return result.content if result and result.content else ""
    except Exception as e:
        print(f"Error in Azure OCR process (from_bytes): {e}")
        st.warning(f"Azure OCR (bytes) failed: {e}")
        return ""

def ocr_with_azure_document_intelligence_from_path(client: DocumentIntelligenceClient, file_path: str) -> str:
    if not client:
        print(f"Azure DI client not available for OCR from path: {file_path}")
        return ""
    try:
        print(f"Attempting Azure OCR from path: {file_path}")
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type is None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".pdf":
                content_type = "application/pdf"
            elif ext in [".jpg", ".jpeg"]:
                content_type = "image/jpeg"
            elif ext == ".png":
                content_type = "image/png"
            else:
                content_type = "application/octet-stream" 
        print(f"Guessed content type for {file_path}: {content_type}")

        with open(file_path, "rb") as f:
            poller = client.begin_analyze_document(
                "prebuilt-read",
                f,
                content_type=content_type
            )
            result = poller.result()
        print(f"Azure OCR from path successful for {file_path}.")
        return result.content if result and result.content else ""
    except Exception as e:
        print(f"Error during Azure OCR process (from_path for {file_path}): {e}")
        st.warning(f"Azure OCR (path: {os.path.basename(file_path)}) failed: {e}")
        return ""


def extract_text_from_pdf(pdf_path: str,
                          azure_client: DocumentIntelligenceClient) -> str:
    """
    • First try pdfplumber on every page.
    • If *anything* goes wrong – or the PDF contains no selectable text –
      fall back to Azure Document Intelligence for the *entire* PDF.
    • No per-page image conversion is attempted anymore.
    """
    if not azure_client: 
        print("Azure DI client unavailable – cannot OCR PDF fallback.")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_chunks = [page.extract_text(x_tolerance=1, y_tolerance=3) for page in pdf.pages if page.extract_text(x_tolerance=1, y_tolerance=3)]
                if text_chunks:
                    print(f"pdfplumber extracted text from {pdf_path} (Azure client was unavailable).")
                    return "\n\n".join(filter(None, text_chunks))
                else:
                    print(f"pdfplumber found no text in {pdf_path} and Azure client is unavailable for fallback.")
                    return ""
        except Exception as err_plumber:
            print(f"pdfplumber error on {pdf_path}: {err_plumber}. Azure client unavailable for fallback.")
            return ""


    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_chunks = []
            for page_num, page in enumerate(pdf.pages, 1):
                txt = page.extract_text(x_tolerance=1, y_tolerance=3)
                if txt and txt.strip():
                    text_chunks.append(txt)

            if text_chunks:                      
                print(f"pdfplumber extracted text from {pdf_path}")
                return "\n\n".join(text_chunks)

            
            print(f"No text found with pdfplumber in {pdf_path}; "
                  "sending whole PDF to Azure DI …")

    except Exception as err:
        
        print(f"pdfplumber error on {pdf_path}: {err}. "
              "Sending whole PDF to Azure DI …")

    
    return ocr_with_azure_document_intelligence_from_path(azure_client, pdf_path)

def extract_text_from_image(image_path: str, azure_client: DocumentIntelligenceClient) -> str:
    print(f"\nProcessing image: {image_path} with Azure Document Intelligence")
    if not azure_client:
        print("Azure DI client not available for image extraction.")
        return ""

    text = ocr_with_azure_document_intelligence_from_path(azure_client, image_path)
    if text and text.strip():
        print("Successfully extracted text from image using Azure OCR.")
    else:
        print("Azure OCR yielded no text for the image.")
    return text


def extract_text_from_txt(txt_path: str) -> str:
    print(f"\nReading text from: {txt_path}")
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading text file {txt_path}: {e}")
        st.warning(f"Could not read text file {os.path.basename(txt_path)}: {e}")
        return ""


def extract_text(file_path: str, azure_client: DocumentIntelligenceClient) -> str:
    """
    Main dispatcher function for text extraction based on file type.
    Uses the provided Azure Document Intelligence client.
    """
    ext = os.path.splitext(file_path)[1].lower()
    extracted_text = ""
    print(f"\nStarting extraction for file: {file_path}, type: {ext}")
    

    if not azure_client and ext not in ['.txt']:
        st.error(f"Azure Document Intelligence client is not available. Cannot process {os.path.basename(file_path)}.")
        print(f"Azure DI client not available, cannot process {file_path}")
        return ""
    elif ext == '.pdf':
        extracted_text = extract_text_from_pdf(file_path, azure_client)
    elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.heic', '.webp']:
        extracted_text = extract_text_from_image(file_path, azure_client)
    elif ext == '.txt':
        extracted_text = extract_text_from_txt(file_path)
    else:
        print(f"Unsupported file type: {ext} for dedicated extractors. Attempting generic Azure OCR for {file_path}.")
        
        if azure_client:
            try:
                extracted_text = ocr_with_azure_document_intelligence_from_path(azure_client, file_path)
                if extracted_text and extracted_text.strip():
                     print(f"Successfully extracted text from unsupported file type {ext} ({file_path}) using Azure OCR.")
                else:
                    print(f"Azure OCR on unsupported file type {ext} ({file_path}) also yielded no text.")
            except Exception as azure_e:
                print(f"Error using Azure DI for unsupported file type {ext} ({file_path}): {azure_e}")
                
                extracted_text = ""
        else:
            print(f"Azure client not available for unsupported file type {ext}.")
           
    
    return extracted_text