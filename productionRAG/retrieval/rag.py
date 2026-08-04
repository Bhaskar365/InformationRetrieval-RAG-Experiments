
from langchain_classic.chains import RetrievalQA
from langchain_ollama import ChatOllama
from retrieval.retriever import get_retriever

llm = ChatOllama(model="llama3")

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=get_retriever()
)

def ask(question:str):

    return rag_chain.invoke(
        { "query": question }
    )

