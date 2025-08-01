from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from groq import Groq

# collecting credentials and connecting to Groq API

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# for loadding all the vectorstores of all domains

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

# for asking query from llm using RAG

def ask(query: str, vectorstore):

    docs = vectorstore.similarity_search(query, k=2)
    context = "\n".join([doc.page_content for doc in docs])

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.5,
        messages=[
            {"role": "system", "content": "You are Maya, a professional AI calling assistant for FinCove Pvt. Ltd., a digital banking platform. You assist users over phone calls with queries related to FinCove’s banking products and services. Answer concisely and clearly using the context"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content