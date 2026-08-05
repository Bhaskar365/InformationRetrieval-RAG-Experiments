
from langchain_chroma import Chroma

from ingestion.embed import embeddings

DB_PATH = "vector_db"


def create_vectorstore(documents):

    print(f"Saving {len(documents)} chunks")

    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print("Vector database created...")

    return db

def load_vectorstore():

    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )


