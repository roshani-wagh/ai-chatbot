import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from fastapi import UploadFile
from io import BytesIO
from langchain_community.vectorstores import FAISS
import faiss
from langchain_openai import OpenAIEmbeddings
from ingest import ingest_documents, create_vector_store_with_retry
import os

s3_bucket_name = os.getenv("S3_BUCKET_NAME")
if not s3_bucket_name:
    raise ValueError("S3_BUCKET_NAME is not set in the environment")

s3_client = boto3.client('s3')
FAISS_INDEX_PATH = "/tmp/faiss_index"  # Temporary path before uploading to S3
FAISS_INDEX_FILE = "faiss_index/index.faiss"

async def upload_document_to_s3(file: UploadFile):
    try:
        # ✅ Upload file to S3
        s3_client.upload_fileobj(file.file, s3_bucket_name, file.filename)

        # ✅ Generate S3 file URL instead of downloading
        file_url = f"https://{s3_bucket_name}.s3.amazonaws.com/{file.filename}"

        # ✅ Process the file directly from S3 URL
        chunks = ingest_documents(file_url)
        vector_store = create_vector_store_with_retry(chunks, api_key=os.getenv("OPENAI_API_KEY"))

        faiss_index_bytes = save_faiss_index_to_bytes(vector_store)

        # ✅ Upload FAISS index to S3
        s3_client.put_object(Bucket=s3_bucket_name, Key="faiss_index/index.faiss", Body=faiss_index_bytes)
        print("✅ FAISS index saved to S3.")
    except NoCredentialsError:
        return {"message": "AWS credentials not available"}

    except ClientError as e:
        return {"message": f"Failed to upload document: {e}"}

    except Exception as e:
        return {"message": f"Failed to process document: {e}"}

    return {"message": "Document uploaded to S3 and indexed!", "file_url": file_url, "vector_store": vector_store}


def save_faiss_index_to_bytes(vector_store):
    """
    Converts a FAISS index into a writable format and serializes it into bytes.
    """
    faiss_index = vector_store.index  # Get FAISS index

    # 🔥 Convert IndexFlatL2 to a writable format
    if isinstance(vector_store.index, faiss.IndexFlatL2):
        vector_store.index = faiss.IndexIDMap(vector_store.index)  # Make it writable

    faiss.write_index(vector_store.index, "faiss_index.bin")
    # ✅ Now it can be serialized
    index_bytes = faiss.serialize_index(faiss_index)

    # Convert to BytesIO for S3 storage
    index_bytes_io = BytesIO(index_bytes)
    index_bytes_io.seek(0)

    return index_bytes_io.getvalue()

def load_faiss_index():
    """
    Loads FAISS index directly from S3.
    """
    try:
        # ✅ Download FAISS index from S3
        response = s3_client.get_object(Bucket=s3_bucket_name, Key="faiss_index/index.faiss")
        faiss_index_bytes = response['Body'].read()

        # ✅ Load FAISS index from bytes
        index_io = BytesIO(faiss_index_bytes)
        faiss_index = faiss.read_index(index_io)

        # ✅ Wrap FAISS index with Langchain’s FAISS
        vector_store = FAISS(index=faiss_index, embedding_function=get_embeddings())

        print("✅ FAISS index loaded from S3.")
        return vector_store

    except ClientError:
        print("⚠️ No existing FAISS index found in S3. Creating a new one.")
        return None  # No index exists yet, return None

    except Exception as e:
        print(f"⚠️ Error loading FAISS index: {e}")
        return None


def get_embeddings(texts):
    embedding_model = OpenAIEmbeddings()
    return embedding_model.embed_documents(texts)