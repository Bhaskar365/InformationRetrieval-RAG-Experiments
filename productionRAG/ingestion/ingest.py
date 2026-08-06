

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    DirectoryLoader
)

from ingestion.chunk import split_documents
from ingestion.vectorstore import create_vectorstore

from pypdf import PdfReader

reader = []

DIR_LOC = "D:\\mlTesting\\FAISS"

LOADERS = {
     "pdf": PyPDFLoader,
     ".txt": TextLoader,
     ".md": TextLoader,
     ".csv": CSVLoader,
     ".docx": UnstructuredWordDocumentLoader,
     ".doc": UnstructuredWordDocumentLoader,
     ".xlsx": UnstructuredExcelLoader
}

def load_docs(folder=f"{DIR_LOC}\\productionRAG\\docs") -> list[Document]:
    """Fetch LangChain documentation pages as Documents."""
    docs = []
    configs = [
        ("**/*.pdf", PyPDFLoader),
        ("**/*.txt", TextLoader),
        ("**/*.md", TextLoader),
        ("**/*.docx", UnstructuredWordDocumentLoader),
        ("**/*.xlsx", UnstructuredExcelLoader),
    ]

    for pattern, loader in configs:
        loaded_docs = DirectoryLoader(
                Path(folder),
                glob=pattern,
                loader_cls=loader,
                recursive=True,
            ).load()

        for doc in loaded_docs:

           source = Path(doc.metadata['source'])

           doc.metadata = {
               "filename" : source.name,
               "file_type" : source.suffix,
               "page_number" : doc.metadata.get("page", 0) + 1,
               "source" : str(source)
           }

        docs.extend(loaded_docs)

    return docs

# documentText = load_docs()

# print(documentText)


# print(f"Loaded {len(documentText)} documents")

# chunks = split_documents(documentText)

# print(f"Created {len(chunks)} chunks")

# db = create_vectorstore(chunks)

if __name__ == "__main__":

    documentText = load_docs()

    print(f"Loaded {len(documentText)} documents")

    chunks = split_documents(documentText)

    print(f"Created {len(chunks)} chunks")

    db = create_vectorstore(chunks)
