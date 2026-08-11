

from retrieval.rag import ask

SESSION_ID = "cli-session"

question = input("\n>>>>>>>>>>>> ")

for event in ask(question, session_id=SESSION_ID):

    if event["type"] == "answer":
        print(event["content"], end="", flush=True)

    elif event["type"] == "sources":
        print("\n\nSources:")

        for source in event["content"]:
            print(
                f"[{source['citation_id']}] "
                f"{source['filename']} "
                f"(page {source['page']})"
            )

print()



# while True:

#     question = input("\n>>>>>>>>>>>> ")

#     if question.lower()== "exit":
#         break

#     # result = ask(question)
#     # print(result)

#     print('\nAnswer: ')

#     sources = []

#     for event in ask(question,session_id=SESSION_ID):

#         # streaming answer

#         if event['type'] == 'answer':

#             print(
#                 event['content'],
#                 end='',
#                 flush=True
#             )

#         # sources arrive after answer

#         elif event['type'] == 'sources':

#             sources = event['content']

#     print('\n')

#     if sources:

#         for source in sources:

#             print(
#                 f"[{source['citation_id']}] "
#                 f"{source['filename']} "
#                 f"Page {source['page']}"
#             )

#     print()

#     # older code
#     # citations = {}

#     # for chunk in ask(question):
#     #     if chunk['type'] == 'token':
#     #         print(chunk, end='', flush=True)

#     #     elif chunk['type'] == 'citations':
#     #         citations = chunk['content']

#     # print("\n")

#     # if citations:
#     #     print('Sources: ')

#     #     for citation_id, source in citations.items():
#     #         print(
#     #             f"[{citation_id}] "
#     #             f"{source['filename']} — "
#     #             f"Page {source['page']}"
#     #         )

#     # print()

#     # answer = result["answer"]

#     # for doc in result["context"]:
#     #     print("=="*100)
#     #     print("Source : ", doc.metadata.get("filename", "Unknown"))
#     #     print("Pages : ", doc.metadata.get("page_number", "Unknown"))
        

