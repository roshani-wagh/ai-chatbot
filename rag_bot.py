from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

def create_rag_bot(vector_store):
    """Creates a Retrieval-Augmented Generation (RAG) bot."""
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # Initialize LLM (Change model if needed)
    llm = OpenAI(temperature=0)

    # Create a QA Chain with source document return
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    
    return qa_chain

def ask_question(qa_chain, query):
    """Processes a user query and returns an answer with sources."""
    if not query.strip():
        return "⚠️ Please provide a valid question.", []

    custom_prompt = f"Please provide a detailed and comprehensive answer to the following question:\n\n{query}"
    
    try:
        response = qa_chain({"query": custom_prompt})
        answer = response.get("result", "⚠️ No answer found.")
        sources = response.get("source_documents", [])

        # Format sources for readability
        source_list = [doc.metadata.get("source", "Unknown") for doc in sources]
        
        return answer, source_list
    except Exception as e:
        return f"⚠️ Error processing query: {str(e)}", []
