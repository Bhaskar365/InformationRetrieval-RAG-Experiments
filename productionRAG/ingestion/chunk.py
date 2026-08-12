
from collections import defaultdict

from langchain_text_splitters import RecursiveCharacterTextSplitter


print("######## LOADED chunk.py ########")
print("CHUNK FILE:", __file__)


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

    print("\n######## ENTERED split_documents ########")

    chunks = splitter.split_documents(documents)

    print("Number of chunks:", len(chunks))

    counters = defaultdict(int)

    for chunk in chunks:

        document_id = chunk.metadata.get(
            "document_id",
            "unknown"
        )

        chunk_number = counters[document_id]

        chunk_id = f"{document_id}_c{chunk_number}"

        print("CREATING:", chunk_id)

        chunk.metadata["chunk_id"] = chunk_id

        counters[document_id] += 1

    print("\n######## AFTER CHUNK ID ASSIGNMENT ########")

    for chunk in chunks[:5]:
        print(chunk.metadata)

    return chunks