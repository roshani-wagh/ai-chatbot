import os
import faiss
import boto3
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
from pdf2image import convert_from_path
import pytesseract
from config import S3_BUCKET_NAME

# Paths
DOCS_DIR = "stored_documents"
FAISS_INDEX_DIR = "faiss_index"
FAISS_INDEX_NAME = "latest.index"
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_DIR, FAISS_INDEX_NAME)

# S3 Client
s3_client = boto3.client("s3")

# Load OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_image_pdf(pdf_path):
    """Extract text from image-based PDFs using OCR."""
    text = ""
    try:
        images = convert_from_path(pdf_path)  # Convert PDF pages to images
        for img in images:
            text += pytesseract.image_to_string(img)  # Extract text using OCR
        return text.strip()
    except Exception as e:
        print(f"⚠️ OCR processing error: {e}")
        return ""

def ingest_documents(file_path: str):
    """
    Ingest documents from a file path and return chunks of text.
    """
    documents = []

    if file_path.endswith(".txt"):
        loader = TextLoader(file_path)
        documents = loader.load()
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # If no text extracted, try OCR
        if not documents or all(not doc.page_content.strip() for doc in documents):
            print(f"⚠️ No text found in {file_path}, using OCR...")
            ocr_text = extract_text_from_image_pdf(file_path)
            if ocr_text:
                documents = [Document(page_content=ocr_text, metadata={"source": file_path})]

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_or_load_faiss():
    """Loads FAISS index if available; otherwise, creates a new one."""
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    if os.path.exists(FAISS_INDEX_PATH):
        print("✅ FAISS index found locally. Loading...")
        return FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    try:
        print("🔹 Checking for FAISS index in S3...")
        temp_file_path = os.path.join(tempfile.gettempdir(), FAISS_INDEX_NAME)
        s3_client.download_file(S3_BUCKET_NAME, FAISS_INDEX_NAME, temp_file_path)

        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        os.rename(temp_file_path, FAISS_INDEX_PATH)

        return FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"⚠️ No FAISS index found in S3. Creating a new one... ({e})")

    # Create new FAISS index
    print("🔹 Creating a new FAISS index...")
    dimension = 1536  # OpenAI embedding dimension
    index = faiss.IndexFlatL2(dimension)  # Empty FAISS index
    vector_store = FAISS(embedding_function=embeddings, index=index, docstore={}, index_to_docstore_id={})

    vector_store.save_local(FAISS_INDEX_DIR)
    print("✅ Saved empty FAISS index locally.")

    return vector_store

def create_vector_store_with_retry(chunks, api_key):
    """Creates or updates the FAISS vector store with error handling."""
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    try:
        if not chunks:
            raise ValueError("⚠️ No valid text chunks found. Cannot create FAISS index.")

        # Load existing FAISS store or create a new one
        vector_store = create_or_load_faiss()
        print(f"✅ Loaded FAISS store, adding {len(chunks)} chunks...")

        vector_store.add_documents(chunks)

        # Save FAISS index locally
        vector_store.save_local(FAISS_INDEX_DIR)
        print("✅ FAISS index updated and saved locally!")

        # Upload FAISS index to S3
        with open(FAISS_INDEX_PATH, "rb") as buffer:
            s3_client.upload_fileobj(buffer, S3_BUCKET_NAME, FAISS_INDEX_NAME)
        print("✅ FAISS index uploaded to S3!")

        return vector_store
    except Exception as e:
        print(f"⚠️ Error updating FAISS index: {e}")

        # Try creating a new FAISS index if updating fails
        try:
            print(f"🔹 Creating a new FAISS index with {len(chunks)} chunks...")
            vector_store = FAISS.from_documents(chunks, embeddings)

            # Save FAISS index locally
            vector_store.save_local(FAISS_INDEX_DIR)
            print("✅ New FAISS index created and saved locally!")

            # Upload FAISS index to S3
            with open(FAISS_INDEX_PATH, "rb") as buffer:
                s3_client.upload_fileobj(buffer, S3_BUCKET_NAME, FAISS_INDEX_NAME)
            print("✅ New FAISS index uploaded to S3!")

            return vector_store
        except Exception as e2:
            print(f"⚠️ Critical error creating FAISS store: {e2}")
            raise RuntimeError(f"Failed to create FAISS vector store. Root cause: {e2}")
def load_vector_store():
    """Loads FAISS vector store and returns a retriever."""
    return create_or_load_faiss().as_retriever(search_type="similarity", search_kwargs={"k": 3})
