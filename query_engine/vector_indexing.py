from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

domains = ["banking", "loan", "insurance", "tax", "investment"]

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

for domain in domains:
    loader = TextLoader(f"docs\{domain}.txt", encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local(f"faiss_index/{domain}")
    print(f"FAISS vectorstore saved for {domain}.")