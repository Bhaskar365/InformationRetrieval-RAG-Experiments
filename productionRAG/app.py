
# from langchain_ollama import ChatOllama
# from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_community.document_loaders import (
#     PyPDFLoader,
#     TextLoader,
#     CSVLoader,
#     UnstructuredWordDocumentLoader,
#     UnstructuredExcelLoader,
#     DirectoryLoader
# )

# from langchain_core.documents import Document
# from pathlib import Path

# DIR_LOC = "D:\\mlTesting\\FAISS"

# LOADERS = {
#      "pdf": PyPDFLoader,
#      ".txt": TextLoader,
#      ".md": TextLoader,
#      ".csv": CSVLoader,
#      ".docx": UnstructuredWordDocumentLoader,
#      ".doc": UnstructuredWordDocumentLoader,
#      ".xlsx": UnstructuredExcelLoader
# }

# def load_docs(folder:str) -> list[Document]:
#     """Fetch LangChain documentation pages as Documents."""
#     docs = []
#     configs = [
#         ("**/*.pdf", PyPDFLoader),
#         ("**/*.txt", TextLoader),
#         ("**/*.md", TextLoader),
#         ("**/*.docx", UnstructuredWordDocumentLoader),
#         ("**/*.xlsx", UnstructuredExcelLoader),
#     ]

#     for pattern, loader in configs:
#         docs.extend(
#             DirectoryLoader(
#                 Path(folder),
#                 glob=pattern,
#                 loader_cls=loader,
#                 recursive=True,
#             ).load()
#         )
#     return docs

# documentText = load_docs(f"{DIR_LOC}\\productionRAG\\docs")

# print(documentText)


from retrieval.rag import ask

while True:

    question = input("> ")

    if question=="exit":
        break

    answer = ask(question)

    print(answer["result"])


