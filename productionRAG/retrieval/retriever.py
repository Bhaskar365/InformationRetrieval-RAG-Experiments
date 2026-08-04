

from ingestion.vectorstore import load_vectorstore

def get_retriever():

    db = load_vectorstore()

    return db.as_retriever(
        search_kwargs={
            "k" : 5
        }
    )

    