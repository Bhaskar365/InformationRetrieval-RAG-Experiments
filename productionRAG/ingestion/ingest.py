
from pathlib import Path

from langchain_core.documents import Document

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    DirectoryLoader,
)

from ingestion.chunk import split_documents
from ingestion.vectorstore import create_vectorstore


DIR_LOC = Path(r"D:\mlTesting\FAISS")
DOCS_DIR = DIR_LOC / "productionRAG" / "docs"


def make_document_id(filename: str) -> str:

    return Path(filename).stem.lower().replace(" ", "-")


def load_docs(folder=DOCS_DIR) -> list[Document]:

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

            source = Path(doc.metadata["source"])

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

        docs.extend(loaded_docs)

    return docs


if __name__ == "__main__":

    documentText = load_docs()

    print(f"Loaded {len(documentText)} documents")

    # IMPORTANT:
    # This now calls ingestion.chunk.split_documents()
    chunks = split_documents(documentText)

    print(f"Created {len(chunks)} chunks")

    for i, chunk in enumerate(chunks[:5]):

        print(f"\n--- CHUNK {i} ---")

        print("Metadata:")
        print(chunk.metadata)

        print("\nContent:")
        print(chunk.page_content[:300])

    create_vectorstore(chunks)