
from langchain_ollama import ChatOllama

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from retrieval.retriever import get_retriever

from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda
)


from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.8,
    num_predict=1000,
)

contextualize_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Given a chat history and the latest user question,
rewrite the latest question into a standalone question.

Do NOT answer the question.
Only rewrite it if necessary.
"""
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

retriever = get_retriever()

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_prompt,
)

# citations helper

def add_citation_ids(docs):
    """
        Add a citation ID to the actual Documents retrieved
        by the RAG pipeline.
    """
    
    for i, doc in enumerate(docs, start=1):
        doc.metadata['citation_id'] = i

    return docs

# QA

qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant.

Answer ONLY using the supplied context.

Each document in the context has a citation number such as:
[1], [2], [3].

When you use information from a document, include the
corresponding citation number immediately after the statement.

Example:

The company was founded in 1995 [1].

Another example:

Revenue increased by 15% in 2025 [1][2].

Rules:

- Use ONLY information from the supplied context.
- Every factual claim should have a citation when appropriate.
- Only use citation numbers that actually exist in the context.
- NEVER invent citation numbers.
- NEVER invent sources.
- NEVER generate filenames or page numbers yourself.
- If the answer is not present in the context, reply exactly:

"I could not find the answer in the provided documents."

Context:

{context}

"""
    ),
    ("human", "{input}"),
])


# question_answer_chain = create_stuff_documents_chain(
#     llm,
#     qa_prompt,
# )



# rag_chain = create_retrieval_chain(
#     history_aware_retriever,
#     question_answer_chain,
# )

# ---------------------------------------------------------------
# This controls how each Document is inserted into {context}.
#
# citation_id comes from:
# doc.metadata["citation_id"]
# ---------------------------------------------------------------
document_prompt = ChatPromptTemplate.from_template(
    "[{citation_id}]\n{page_content}"
)

# QA chain
question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt,
    document_prompt=document_prompt,
)

retrieve_and_cite = RunnablePassthrough.assign(
        context=history_aware_retriever,
    ).assign(
        context=RunnableLambda(
        lambda x: add_citation_ids(x["context"])
    )
)

rag_chain = retrieve_and_cite.assign(
    answer=question_answer_chain
)

session_id = "cli-session"

store = {}

def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)

# Streaming API

def ask(question, session_id='default'):
    """
    Stream the answer and return citation metadata
    after the answer has finished.

    Events returned:

        {
            "type": "answer",
            "content": "some text"
        }

    and finally:

        {
            "type": "sources",
            "content": [...]
        }
    """

    retrieved_docs = []

    for chunk in conversational_rag_chain.stream(
        { "input": question },

        config={ 
            "configurable": { 
                "session_id": session_id 
                }
            }
    ):
        if 'answer' in chunk:
            answer_chunk = chunk['answer']
            if answer_chunk:
                yield{
                    'type': 'answer',
                    'content': answer_chunk
                }


    # collect info and create information source
    sources = []

    for doc in retrieved_docs:

        citation_id = doc.metadata.get(
            'citation_id',
            '?'
        )

        sources.append({
            'citation_id': citation_id,

            'filename': doc.metadata.get(
                'filename',
                'Unknown'
            ),

            'page': doc.metadata.get(
                'page_number',
                'Unknown'
            ),
        })


        yield {
            'type': 'sources',
            'content': sources
        }


# Non streaming API
def ask_full(question, session_id='default'):

    """
    Use this when you don't need streaming.

    Returns:

    {
        "input": ...,
        "context": [...],
        "answer": "..."
    }
    """

    return conversational_rag_chain.invoke(
        {"input": question},

        config={
            "configurable": {
                "session_id": session_id
            }
        },
    )


# Older code

# def ask(question, session_id='default'):
#     return conversational_rag_chain.stream(
#         { "input": question },
#         config={ "configurable": { "session_id": session_id } }
#     )

# def ask(questions, session_id='default'):
#     for chunk in conversational_rag_chain.stream(
#         {"input": questions},
#         config={"configurable": {"session_id": session_id}}
#     ):
#         if "answer" in chunk:
#             yield chunk["answer"]
