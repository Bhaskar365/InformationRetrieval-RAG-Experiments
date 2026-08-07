

from retrieval.rag import ask

while True:

    question = input("> ")

    if question=="exit":
        break

    result = ask(question)

    print(result)

    answer = result["answer"]
    print(answer)

    for doc in result["context"]:
        print("=="*100)
        print("Source : ", doc.metadata.get("source", "Unknown"))
        print("Pages : ", doc.metadata.get("page", "Unknown"))
        

