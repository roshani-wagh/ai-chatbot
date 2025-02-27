import os
from io import StringIO

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import logging
import boto3
import csv
import difflib
import rapidfuzz
from fuzzywuzzy import process, fuzz

logger = logging.getLogger(__name__)

s3_client = boto3.client("s3")

def load_csv_from_s3(bucket_name, csv_key):
    file_url_map = {}

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=csv_key)
        csv_content = response["Body"].read().decode("utf-8")

        csv_reader = csv.DictReader(StringIO(csv_content))
        for row in csv_reader:
            file_url_map[row["filename"]] = row["url"]

        logger.info(f"Loaded {len(file_url_map)} entries from {csv_key}.")

    except Exception as e:
        logger.error(f"Error reading CSV from S3: {e}")

    return file_url_map

def create_rag_bot(vector_store):

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
    
    prompt_template = """You are an expert knowledge assistant. Use the following pieces of retrieved context to answer the question thoroughly and accurately.

Context:
{context}

Question: {question}

Instructions:
-Answer in complete and concise sentences.
-Strictly categorize the question before responding:
    If the question is related to self-harm, respond with:
    "I detected a statement that you intend to cause yourself harm. Please visit https://help.ea.com/en for any help."
    If the question contains toxic or inappropriate language or doesnt match context, respond with:
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
-Retrieve and reference all relevant parts of the uploaded PDFs before answering. Think step by step and provide a detailed response.
Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

static_answers = {
            "refund ": "Find out how to get a refund for games that qualify under the great game guarantee policy here: \nhttps://help.ea.com/en-us/help/account/returns-and-cancellations/",
            "delete account": "To delete your Ea account, visit https://help.ea.com/en/help/account/close-ea-account",
            "reset password": "To reset your password, visit https://ea.com/reset-password",
            "crafting legend tokens": "Find out more about Crafting Metals and Legend Tokens on this site \nhttps://help.ea.com/in/solutions/?product=apex-legends&platform=&topic=metals-tokens-info",
            "crafting metals" : "Find out more about Crafting and Legend Tokens on this site \nhttps://help.ea.com/in/solutions/?product=apex-legends&platform=&topic=metals-tokens-info",
            "reset rank" : "Please refer this link to get more help on resetting rank \nhttps://help.ea.com/in/solutions/?product=apex-legends&topic=emerging-technical-support-09",
            "progress restore" : "Check out what you can do to get back in the game here: \nhttps://help.ea.com/in/help/apex-legends/apex-legends/game-progress",
        }

def get_static_answer(question):
    best_match = process.extractOne(question.lower(), static_answers.keys(), scorer=fuzz.token_sort_ratio)

    if best_match:
        matched_key, score = best_match[0], best_match[1]  # Extract matched string and score

        if score > 65:  # Adjust threshold as needed
            return static_answers[matched_key]

    return None

def ask_question(qa_chain, question, file_urls):
    static_answer = get_static_answer(question)

    if static_answer:
        return static_answer, []

    if not question.strip():
        return "⚠️ Please provide a valid question.", []

    try:
        logger.info(f"Processing query: {question}")

        # Check if question matches any static answer
        for key, answer_text in static_answers.items():
            if key in question.lower():
                return answer_text, []  # Return static answer immediately

        expected_keys = getattr(qa_chain, 'input_keys', ['unknown'])
        logger.info(f"Chain expects these input keys: {expected_keys}")

        if 'question' in expected_keys:
            response = qa_chain({"question": question})
        elif 'query' in expected_keys:
            response = qa_chain({"query": question})
        else:
            input_param = {expected_keys[0]: question}
            response = qa_chain(input_param)

        if isinstance(response, dict):
            answer = response.get("result") or response.get("answer") or "⚠️ No answer found in response."

            non_contextual_responses = {
                "I’m sorry, I can’t provide a response, as your question appears to be inappropriate and not related to EA.",
                "Hi, how can I help you? Please ask me a question related to EA.",
                "Goodbye! If you have any more questions in the future, feel free to ask. Have a great day.",
                "I detected a statement that you intend to cause yourself harm. Please visit https://help.ea.com/en for any help."
            }

            if answer in non_contextual_responses or "not sure how to answer" in answer.lower():
                return answer, []

            rejection_patterns = [
                "I’m sorry, I can’t provide a response",
                "I am unable to answer that",
                "This question is inappropriate"
            ]

            if any(pattern in answer for pattern in rejection_patterns):
                return answer, []

            sources = response.get("source_documents", [])
            source_list = set()
            for doc in sources:
                source_path = os.path.basename(doc.metadata.get("source", "Unknown"))
                if source_path in file_urls:
                    source_list.add(file_urls[source_path])

            if source_list:
                sources_text = "\n\n Learn more at:\n" + "\n".join(source_list)
                answer += sources_text

            logger.info(f"Successfully generated answer: {answer[:100]}...")
            logger.info(f"Sources: {source_list}")

            return answer, list(source_list)

        else:
            logger.error(f"Unexpected response type: {type(response)}")
            return "⚠️ Received an unexpected response format.", []

    except Exception as e:
        logger.error(f"⚠️ Error processing query: {e}", exc_info=True)
        return f"⚠️ Error processing query: {str(e)}", []
