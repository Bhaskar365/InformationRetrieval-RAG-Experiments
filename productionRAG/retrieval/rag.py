
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


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
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


qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant.

Answer ONLY using the supplied context.

If the answer is not present, reply:

"I could not find the answer in the provided documents."

Context:
{context}
"""
    ),
    ("human", "{input}"),
])


history_aware_retriever = create_history_aware_retriever(
    llm,
    get_retriever(),
    contextualize_prompt,
)


question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt,
)

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain,
)

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

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

# def ask(question, session_id='default'):
#     return conversational_rag_chain.stream(
#         { "input": question },
#         config={ "configurable": { "session_id": session_id } }
#     )

def ask(questions, session_id='default'):
    for chunk in conversational_rag_chain.stream(
        {"input": questions},
        config={"configurable": {"session_id": session_id}}
    ):
        if "answer" in chunk:
            yield chunk["answer"]
