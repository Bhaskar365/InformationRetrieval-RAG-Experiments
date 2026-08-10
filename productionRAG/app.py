

from retrieval.rag import ask

while True:

    question = input("> ")

    if question=="exit":
        break

    # result = ask(question)
    # print(result)

    for text in ask(question):
        print(text, end='', flush=True)

    # answer = result["answer"]

    # for doc in result["context"]:
    #     print("=="*100)
    #     print("Source : ", doc.metadata.get("filename", "Unknown"))
    #     print("Pages : ", doc.metadata.get("page_number", "Unknown"))
        

