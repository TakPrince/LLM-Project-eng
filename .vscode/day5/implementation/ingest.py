import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # 🔁 Changed to free local embeddings

from dotenv import load_dotenv

# Path setup
BASE_DIR = Path(__file__).parent.parent
DB_NAME = str(BASE_DIR / "vector_db")
KNOWLEDGE_BASE = str(BASE_DIR / "knowledge-base")

load_dotenv(override=True)

# 1. Replace OpenAI with Free Local Embeddings
# 'all-MiniLM-L6-v2' is small, fast, and very effective for RAG.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def fetch_documents():
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder, 
            glob="**/*.md", 
            loader_cls=TextLoader, 
            loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents

def create_chunks(documents):
    # Balanced settings for local embedding models
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100 
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_embeddings(chunks):
    # Clean up old DB if it exists to avoid dimension mismatch
    if os.path.exists(DB_NAME):
        import shutil
        shutil.rmtree(DB_NAME)

    # Initialize Chroma with the free local embeddings
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_NAME
    )

    count = vectorstore._collection.count()
    print(f"Ingestion Complete: {count:,} chunks stored locally in {DB_NAME}")
    return vectorstore

if __name__ == "__main__":
    print("Fetching documents...")
    docs = fetch_documents()
    print(f"Loaded {len(docs)} documents. Splitting into chunks...")
    chunk_list = create_chunks(docs)
    print(f"Created {len(chunk_list)} chunks. Generating embeddings locally...")
    create_embeddings(chunk_list)