from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import logging
from langchain.chains.question_answering import load_qa_chain

logger = logging.getLogger(__name__)

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
- Answer in complete sentences
- If the context doesn't contain relevant information, say "I don't have enough information to answer this question."
- Provide specific details from the context when available
- Be concise but comprehensive

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

def ask_question(qa_chain, query):
    """Processes a user query with improved error handling and consistent output."""
    if not query.strip():
        return "⚠️ Please provide a valid question.", []
    
    try:
        logger.info(f"Processing query: {query}")
        
        # Debug the expected input keys
        expected_keys = getattr(qa_chain, 'input_keys', ['unknown'])
        logger.info(f"Chain expects these input keys: {expected_keys}")
        
        # Try with the most common parameter names
        if 'question' in expected_keys:
            response = qa_chain({"question": query})
        elif 'query' in expected_keys:
            response = qa_chain({"query": query})
        else:
            # Default fallback using the first expected key
            input_param = {expected_keys[0]: query}
            response = qa_chain(input_param)
        
        # Extract answer with better fallback handling
        if isinstance(response, dict):
            answer = response.get("result") or response.get("answer") or "⚠️ No answer found in response."
            
            # Extract sources with better fallback handling
            sources = response.get("source_documents", [])
            if not sources and 'sources' in response:
                sources_text = response.get("sources", "")
                return answer, [sources_text] if sources_text else []
            
            source_list = [doc.metadata.get("source", "Unknown") for doc in sources]
            
            logger.info(f"Successfully generated answer: {answer[:100]}...")
            logger.info(f"Sources: {source_list}")
            
            return answer, source_list
        else:
            logger.error(f"Unexpected response type: {type(response)}")
            return "⚠️ Received an unexpected response format.", []
        
    except Exception as e:
        logger.error(f"⚠️ Error processing query: {e}", exc_info=True)
        return f"⚠️ Error processing query: {str(e)}", []