

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "Give me 3 career skills that are in high demand in {year}."
)

llm = ChatOllama(model='llama3.2:3b')

chain = prompt | llm | StrOutputParser()

response = chain.invoke({"year":"2026,2025,2024"})

print(response)

#  parser streaming
parser = StrOutputParser()

# With streaming - use transform() to process a stream
stream = llm.stream("Tell me a story")
for chunk in parser.transform(stream):
    print(chunk, end="", flush=True)


