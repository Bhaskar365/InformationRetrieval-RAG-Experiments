
    
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
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

reader = []

# DIR_LOC = "D:\\mlTesting\\FAISS"

DIR_LOC = Path(r"D:\mlTesting\FAISS")
DOCS_DIR = DIR_LOC / "productionRAG" / "docs"

LOADERS = {
     "pdf": PyPDFLoader,
     ".txt": TextLoader,
     ".md": TextLoader,
     ".csv": CSVLoader,
     ".docx": UnstructuredWordDocumentLoader,
     ".doc": UnstructuredWordDocumentLoader,
     ".xlsx": UnstructuredExcelLoader
}


splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

def make_document_id(filename: str) -> str:
    return Path(filename).stem.lower().replace(" ", "-")

def split_documents(documents):
    return splitter.split_documents(documents)


def load_docs(folder=DOCS_DIR) -> list[Document]:
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

           filename = source.name
           document_id = make_document_id(filename)

           original_page = doc.metadata.get("page", 0)

           doc.metadata = {
                "document_id": document_id,
                "filename": filename,
                "file_type": source.suffix.lower(),
                "page_number": original_page + 1,
                "source": str(source),
            }

        # old metadata

        #    doc.metadata = {
        #        "filename" : source.name,
        #        "file_type" : source.suffix,
        #        "page_number" : doc.metadata.get("page", 0) + 1,
        #        "source" : str(source)
        #    }

        docs.extend(loaded_docs)

    return docs

if __name__ == "__main__":

    documentText = load_docs()

    print(f"Loaded {len(documentText)} documents")

    chunks = split_documents(documentText)

    print(f"Created {len(chunks)} chunks")

    for i, chunk in enumerate(chunks[:5]):

        print(f"\n--- CHUNK {i} ---")

        print("Metadata:")
        print(chunk.metadata)

        print("\nContent:")
        print(chunk.page_content[:300])

    # print(f"Created {len(chunks)} chunks")

    # for chunk in chunks[:5]:

    #     print("\n--- CHUNK ---")

    #     print(chunk.metadata)

    #     print(
    #         chunk.page_content[:300]
    #     )

    db = create_vectorstore(chunks)

# Older code


# documentText = load_docs()

# print(documentText)


# print(f"Loaded {len(documentText)} documents")

# chunks = split_documents(documentText)

# print(f"Created {len(chunks)} chunks")

# db = create_vectorstore(chunks)
