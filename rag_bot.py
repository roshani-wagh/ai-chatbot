import os
from io import StringIO

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import logging
import boto3
import csv
from langchain.chains.question_answering import load_qa_chain

from config import S3_BUCKET_NAME

logger = logging.getLogger(__name__)

s3_client = boto3.client("s3")

def load_csv_from_s3(bucket_name, csv_key):
    """Downloads and parses a CSV from S3, returning a dictionary mapping filenames to URLs."""
    file_url_map = {}

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=csv_key)
        csv_content = response["Body"].read().decode("utf-8")

        # Read CSV content into dictionary
        csv_reader = csv.DictReader(StringIO(csv_content))
        for row in csv_reader:
            file_url_map[row["filename"]] = row["url"]

        logger.info(f"Loaded {len(file_url_map)} entries from {csv_key}.")

    except Exception as e:
        logger.error(f"Error reading CSV from S3: {e}")

    return file_url_map

def create_rag_bot(vector_store):
    """Creates an improved Retrieval-Augmented Generation (RAG) bot with better answer quality."""
    
    # Increase k to retrieve more relevant documents
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    # Use a more capable model with slightly higher temperature for more detailed responses
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
    
    # Create a comprehensive prompt that encourages detailed answers
    prompt_template = """You are an expert knowledge assistant. Use the following pieces of retrieved context to answer the question thoroughly and accurately.

Context:
{context}

Question: {question}

Instructions:
-Answer in complete and concise sentences.
-Strictly categorize the question before responding:
    If the question is related to self-harm, respond with:
    "I detected a statement that you intend to cause yourself harm. Please visit https://help.ea.com/en for any help."
    If the question contains toxic or inappropriate language, respond with:
    "I’m sorry, I can’t provide a response, as your question appears to be inappropriate and not related to EA."
    If the question is a greeting (e.g., "hi," "hello," "how are you?"), respond with:
    "Hi, how can I help you? Please ask me a question related to EA."
    If the question is a goodbye message (e.g., "bye," "see you later"), respond with:
    "Goodbye! If you have any more questions in the future, feel free to ask. Have a great day."
-If the question is unrelated to EA and does not fit the above categories, respond with:
    "I'm not sure how to answer that. Please ask a question related to EA." (Do not use this phrase verbatim.)
-If the question is relevant to EA and has contextual information available, follow these rules:
    Answer based only on the given context.
    Provide a detailed response with at least 5 numbered points (unless unnecessary for short, direct answers).
    Include specific details from the provided context.
    If the information comes from an article, provide a source link from the "Learn More" section at the end.
Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create custom QA chain with the refactored prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain



def ask_question(qa_chain, question, file_urls):

    file_urls= file_urls
    if not question.strip():
        return "⚠️ Please provide a valid question.", []
    
    try:
        logger.info(f"Processing query: {question}")
        
        # Debug the expected input keys
        expected_keys = getattr(qa_chain, 'input_keys', ['unknown'])
        logger.info(f"Chain expects these input keys: {expected_keys}")
        
        # Try with the most common parameter names
        if 'question' in expected_keys:
            response = qa_chain({"question": question})
        elif 'query' in expected_keys:
            response = qa_chain({"query": question})
        else:
            # Default fallback using the first expected key
            input_param = {expected_keys[0]: question}
            response = qa_chain(input_param)
        
        # Extract answer with better fallback handling
        if isinstance(response, dict):
            answer = response.get("result") or response.get("answer") or "⚠️ No answer found in response."
            
            # Extract sources with better fallback handling
            sources = response.get("source_documents", [])
            if not sources and 'sources' in response:
                sources_text = response.get("sources", "")
                return answer, [sources_text] if sources_text else []

            source_list = set()
            for doc in sources:
                # source_path = doc.metadata.get("source", "Unknown").replace("/tmp/", "")
                source_path = os.path.basename(doc.metadata.get("source", "Unknown"))
                source_list.add(file_urls.get(source_path, source_path))


            logger.info(f"Successfully generated answer: {answer[:100]}...")
            logger.info(f"Sources: {source_list}")
            
            return answer, list(source_list)
        else:
            logger.error(f"Unexpected response type: {type(response)}")
            return "⚠️ Received an unexpected response format.", []
        
    except Exception as e:
        logger.error(f"⚠️ Error processing query: {e}", exc_info=True)
        return f"⚠️ Error processing query: {str(e)}", []