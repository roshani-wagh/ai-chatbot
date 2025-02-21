from fastapi import FastAPI, UploadFile, Request,HTTPException
import shutil
from pydantic import BaseModel
import os
import boto3
import logging
import tempfile
from ingest import ingest_documents, create_vector_store_with_retry, load_vector_store
from rag_bot import create_rag_bot, ask_question
from config import S3_BUCKET_NAME
# Add Pydantic model for request
class QuestionRequest(BaseModel):
    query: str

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Your Question here"
            }
        }
# Add Pydantic model for response
class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]

app = FastAPI()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3_client = boto3.client("s3")

vector_store = None
index_path = "./faiss_indexes/latest.index"

@app.on_event("startup")
async def startup_event():
    global vector_store
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".index") as temp_file:
            s3_client.download_file(S3_BUCKET_NAME, "latest.index", temp_file.name)
            temp_file_path = temp_file.name

        vector_store = load_vector_store()
        os.remove(temp_file_path)
        print("✅ Loaded existing vector store from S3")
    except Exception as e:
        print(f"⚠️ No FAISS index found in S3: {e}. Upload a document first.")

@app.post("/upload/")
async def upload_document(file: UploadFile):
    global vector_store

    try:
        temp_file_path = os.path.join(tempfile.gettempdir(), file.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Uploading {file.filename} to S3...")
        with open(temp_file_path, "rb") as buffer:
            s3_client.upload_fileobj(buffer, S3_BUCKET_NAME, file.filename)
        logger.info("✅ Upload successful!")

        logger.info(f"Processing file {file.filename} for vector store...")
        chunks = ingest_documents(temp_file_path)
        vector_store = create_vector_store_with_retry(chunks, os.getenv("OPENAI_API_KEY"))

        return {"message": "File uploaded & indexed successfully!", "index_name": "latest.index"}
    except Exception as e:
        logger.error(f"Failed to upload document: {e}")
        return {"message": f"Failed to upload document: {e}"}

@app.post("/ask/", response_model=QuestionResponse)
async def ask(question: QuestionRequest):
    """
    Ask a question about the uploaded documents.
    
    The query will be processed using RAG (Retrieval-Augmented Generation) to provide
    relevant answers based on the uploaded document content.
    """
    global vector_store

    if vector_store is None:
        raise HTTPException(status_code=400, detail="⚠️ No documents available. Please upload a document first.")

    try:
        # Use RAG bot for better answer generation
        qa_chain = create_rag_bot(vector_store)
        answer, sources = ask_question(qa_chain, question.query)

        return QuestionResponse(answer=answer, sources=sources)
    
    except Exception as e:
        logger.error(f"⚠️ Internal Server Error: {e}")
        raise HTTPException(status_code=500, detail=f"⚠️ Internal Server Error: {str(e)}")