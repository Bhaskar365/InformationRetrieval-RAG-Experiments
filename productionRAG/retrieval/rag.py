
from langchain_classic.chains import RetrievalQA
from langchain_ollama import ChatOllama
from retrieval.retriever import get_retriever

from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model="llama3.2:3b", temperature=0, num_predict=1000)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.
Answer ONLY using the supplied context.
If the answer is not present, reply:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
)

rag_chain = None

def get_rag_chain():

    global rag_chain

    if rag_chain is None:
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=get_retriever(),
            chain_type='stuff',
            chain_type_kwargs={
                'prompt': prompt
            },
            return_source_documents=True
        )

    return rag_chain

def ask(question):

    chain = get_rag_chain()

    return chain.invoke({ "query": question })

