import tempfile
import boto3
import os
import asyncio
from ingest import ingest_documents, create_vector_store_with_retry

s3_client = boto3.client("s3")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")


async def process_document(filename: str):
    try:
        # Get file from S3
        s3_object = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=filename)
        file_stream = s3_object["Body"].read()

        # 🔹 Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_stream)
            temp_file_path = temp_file.name

        chunks = ingest_documents(temp_file_path)
        vector_store = await asyncio.to_thread(create_vector_store_with_retry, chunks, os.getenv("OPENAI_API_KEY"))
        os.remove(temp_file_path)

        return vector_store
    except Exception as e:
        print(f"Error processing document {filename}: {e}")
        return None
