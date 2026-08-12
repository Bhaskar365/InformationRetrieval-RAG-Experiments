

from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter



splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

def split_documents(documents):
    # return splitter.split_documents(documents)

    

    chunks = splitter.split_documents(documents)

    print(
        f"\nDEBUG: splitter created {len(chunks)} chunks"
    )

    counters = defaultdict(int)

    for chunk in chunks:

        document_id = chunk.metadata.get(
            "document_id",
            "unknown",
        )

        chunk_number = counters[document_id]
        counters[document_id] += 1

        chunk.metadata["chunk_id"] = (
            f"{document_id}_c{chunk_number}"
        )

    return chunks



print(
    "USING split_documents FROM:",
    split_documents.__code__.co_filename
)