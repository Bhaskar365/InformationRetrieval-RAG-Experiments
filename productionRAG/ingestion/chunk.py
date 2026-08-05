

from langchain_text_splitters import RecursiveCharacterTextSplitter

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

def split_documents(documents):
    return splitter.split_documents(documents)


