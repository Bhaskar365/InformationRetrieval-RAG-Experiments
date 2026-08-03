
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_classic.chains import RetrievalQA

doc_text = """
Elon Musk is a technology entrepreneur and engineer known for founding SpaceX and Tesla.
He was born on June 28, 1971, in Pretoria, South Africa.
His major achievements include advancing space exploration and electric vehicles.
Musk is also involved with Neuralink and The Boring Company.
This document provides a brief overview of Musk's background and accomplishments.
"""

document = Document(page_content=doc_text, metadata={"source": "in-memory-doc"})

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_split = text_splitter.split_documents([document])

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(docs_split, embeddings)

llm = ChatOllama(model='llama3.2:3b')

retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type='stuff'
)

query = "Who is Elon Musk and what are his major achievements?"

result = qa_chain.invoke({ "query" : query })

print("Answer:", result['result'])
print("\nSource Documents:")
for doc in result['source_documents']:
    print(f"- {doc.page_content}")


