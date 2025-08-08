from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
import os
from typing import Dict, List

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0.5)

# Store session history per SID
session_store: Dict[str, BaseChatMessageHistory] = {}

class SlidingWindowChatMessageHistory(BaseChatMessageHistory):
    """Simple sliding window approach - keeps only recent messages"""
    
    def __init__(self, window_size: int = 16):
        self.window_size = window_size
        self._messages: List[BaseMessage] = []
    
    @property
    def messages(self) -> List[BaseMessage]:
        return self._messages
    
    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)
        
        if len(self._messages) > self.window_size:
            self._messages = self._messages[-self.window_size:]
    
    def clear(self) -> None:
        self._messages = []

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create session memory for a given session ID"""
    if session_id not in session_store:
        session_store[session_id] = SlidingWindowChatMessageHistory(window_size=16)
    return session_store[session_id]

# Load all domain-specific FAISS vectorstores
def load_all_vectorstores():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    domains = ["banking", "loan", "insurance", "tax", "investment"]
    vector_stores = {}
    
    for domain in domains:
        path = f"query_engine/faiss_index/{domain}"
        try:
            vector_stores[domain] = FAISS.load_local(
                folder_path=path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Loaded vector store for domain: {domain}")
        except Exception as e:
            print(f"Failed to load vector store for domain '{domain}': {e}")
    
    return vector_stores

# Create the conversational RAG chain
def create_conversational_rag_chain(vectorstore):
    
    # Prompt for contextualizing questions based on chat history
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as it is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, vectorstore.as_retriever(), contextualize_q_prompt
    )
    
    # Prompt for answering questions
    qa_system_prompt = """You are Maya, a professional female AI calling assistant for FinCove Pvt. Ltd., \
    a digital banking platform. You assist users over phone calls with queries related to FinCove's \
    banking products and services. Answer concisely and clearly using the provided context.
    Do not use any Markdown formatting like **bold**, *italic*, or `code`. Give only simple text response without any emphasis.
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create question-answering chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create the full RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Add memory to the chain
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_rag_chain

# Global variable to store the chain (initialize once)
conversational_chain = None

def initialize_chain(vectorstore):
    """Initialize the conversational chain once"""
    global conversational_chain
    if conversational_chain is None:
        conversational_chain = create_conversational_rag_chain(vectorstore)
    return conversational_chain

# Ask function using conversational RAG
def ask(SID: str, query: str, vectorstore):
    try:
        # Initialize the chain if not already done
        chain = initialize_chain(vectorstore)
        
        # Invoke the chain with session configuration
        result = chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": SID}}
        )

        response = result["answer"]
        
        if result and "answer" in result:
            return response
        else:
            return "I apologize, but I couldn't find an answer to your question. Please try rephrasing it."
    
    except Exception as e:
        print(f"Error in ask function: {e}")
        return "I apologize, but I encountered an error processing your request. Please try again."