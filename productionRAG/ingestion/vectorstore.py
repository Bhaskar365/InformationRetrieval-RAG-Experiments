
from langchain_chroma import Chroma
from ingestion.embed import embeddings

from dotenv import load_dotenv
import os
import pickle
from pathlib import Path

# DB_PATH = "vector_db"
DB_PATH = Path("vector_db")

load_dotenv()

def create_vectorstore(documents):

    print(f"Saving {len(documents)} chunks")

    DB_PATH.mkdir(parents=True, exist_ok=True)

    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(DB_PATH)
    )
   
    with open(os.getenv("BM25_DOCUMENT_PATH"), "wb") as f:
        pickle.dump(documents, f) 

    print("Vector database created...")
    print("Document corpus saved for BM25...")

    return db

def load_vectorstore():

    return Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=embeddings
    )

# def load_documents():

#     with open(os.getenv("BM25_DOCUMENT_PATH", "rb")) as f:
#         return pickle.load(f)

def load_documents():

    path = os.getenv("BM25_DOCUMENT_PATH")

    if not path:
        raise ValueError(
            "BM25_DOCUMENT_PATH is not set in .env"
        )

    with open(path, "rb") as f:
        return pickle.load(f)

