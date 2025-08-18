import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

#if i have open ai
# os.environ["OPENAI_API_KEY"] = "MY_OPENAI_API_KEY"

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_vector_db():
    """
    This function creates a FAISS vector database from PDF documents
    using a Hugging Face embedding model.
    """
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} document pages.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split documents into {len(texts)} chunks.")

    # This will download the model from the Hugging Face Hub the first time you run it.
    # It runs entirely on local machine (CPU).
    print("Loading Hugging Face embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", model_kwargs={'device': 'cpu'}, show_progress=True)
    print(embeddings)
    
    # 4. Store the embeddings in a FAISS vector store
    print("Creating FAISS vector store...")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector store created and saved at {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()