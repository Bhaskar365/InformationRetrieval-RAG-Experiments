

from retrieval.rag import ask

while True:

    question = input("> ")

    if question=="exit":
        break

    answer = ask(question)

    print(answer["result"])

    for doc in answer["source_documents"]:
        print("=="*100)
        print("Source : ", doc.metadata.get("filename", "Unknown"))
        print("Pages : ", doc.metadata.get("page_number", "Unknown"))


