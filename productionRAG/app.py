

from retrieval.rag import ask

while True:

    question = input("> ")

    if question=="exit":
        break

    answer = ask(question)

    print(answer["result"])

    for doc in answer["source_documents"]:
        print("=="*100)
        print("Creator : ", doc.metadata["creator"])
        print("Total Pages : ", doc.metadata["total_pages"])


