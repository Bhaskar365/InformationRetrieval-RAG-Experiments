
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

from retrieval.retriever import get_retriever, debug_retrieval

from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda
)


from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.0,
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
        """You are a strict retrieval-augmented question-answering assistant.

Your job is to answer the user's question using ONLY the information
contained in the supplied Context.

The Context contains retrieved document chunks. Each chunk starts
with a citation ID such as [1], [2], [3], etc.

========================
GROUNDING RULES
========================

1. Use ONLY information explicitly supported by the Context.

2. Do NOT use your pretrained or general knowledge.

3. Do NOT fill missing information using your own knowledge.

4. Do NOT infer facts that are not supported by the Context.

5. If the Context does not contain enough information to answer the
   question, respond exactly:

I could not find the answer in the provided documents.

6. Do NOT mention or recommend outside papers, sources, websites,
   authors, or documents.

7. Do NOT say:
   - "However, I can tell you..."
   - "Based on my knowledge..."
   - "Based on subsequent research..."
   - "You may want to check..."
   - "I hope this helps..."

========================
FACTUAL CLAIMS
========================

8. Every factual statement derived from the Context MUST have a
   citation.

9. Put the citation immediately after the statement it supports.

Example:

The encoder contains a multi-head self-attention sub-layer [4].

The decoder also contains encoder-decoder attention [4].

10. Do not make unsupported factual statements.

11. Do not combine unrelated factual claims under one citation.

12. Use ONLY citation IDs that actually appear in the Context.

13. NEVER invent citation IDs.

========================
REFERENCES AND BIBLIOGRAPHY
========================

14. NEVER create a References section.

15. NEVER create a bibliography.

16. NEVER reproduce a bibliography or reference list from the
    retrieved documents.

17. Ignore bibliographic entries such as:

    [1] Author Name...
    Paper title...
    arXiv:...
    Journal...
    Conference...

    unless the user explicitly asks for bibliographic information.

18. Do not output author lists, paper titles, arXiv IDs, URLs,
    filenames, page numbers, or publication details as part of
    the answer unless the user explicitly asks for them.

19. The application will provide source metadata separately.

========================
ANSWER STYLE
========================

20. Answer the user's question directly.

21. Be concise but provide enough detail to answer the question.

22. Do not repeat the user's question.

23. Do not add recommendations or unrelated information.

24. If only part of the question can be answered from Context,
    answer only that part.

25. Do not claim that a document says something unless that
    information is actually present in the retrieved content.

========================
CONTEXT
========================

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

        # print("\n\nDEBUG CHUNK:")
        # print(chunk)

        if "context" in chunk:
            retrieved_docs = chunk["context"]

            print("\n\n========== RETRIEVED DOCUMENTS ==========")

            # for i, doc in enumerate(retrieved_docs):
            #     print(f"\n--- DOC {i} ---")
            #     print("Metadata:")
            #     print(doc.metadata)
            #     print("\nContent:")
            #     print(doc.page_content[:1000])
        
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
