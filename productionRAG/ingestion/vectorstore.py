
from langchain_chroma import Chroma

from ingestion.embed import embeddings

DB_PATH = "vector_db"


def create_vectorstore(documents):

    db = Chroma.from_documents(
        documents,
        embeddings=embeddings,
        persist_directory=DB_PATH
    )

    return db

def load_vectorstore():

    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )


