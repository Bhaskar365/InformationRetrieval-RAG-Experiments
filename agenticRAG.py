
from dotenv import load_dotenv
import requests
import os

load_dotenv()

import numpy as np

class VectorDatabase:

    def __init__(self):
        self.vectors = []

    def add_vector(self, vec_id, vector, metadata=None):
        record = {
            "id": vec_id,
            "vector": np.array(vector, dtype=np.float32),
            "metadata": metadata
        }

        self.vectors.append(record)

    def get_all_vectors(self):
        return self.vectors

    def _cosine_similarity(self, vec_a, vec_b):

        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        cos_sim = dot_product / (norm_a * norm_b + 1e-8)

        return cos_sim

    def search(self, query_vector, top_k=3):

        query_vector = np.array(query_vector, dtype=np.float32)
        results = []

        for record in self.vectors:
            sim = self._cosine_similarity(query_vector, record['vector'])

            results.append({
                'id': record['id'],
                'similarity': sim,
                'metadata': record['metadata']
            })
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

db = VectorDatabase()

np.random.seed(10)

db.add_vector('vec1', np.random.rand(5), metadata= { 'text': 'First item' })
db.add_vector('vec2', np.random.rand(5), metadata= { 'text': 'Second item' })
db.add_vector('vec3', np.random.rand(5), metadata= { 'text': 'Third item' })
db.add_vector('vec4', np.random.rand(5), metadata= { 'text': 'Fourth item' })
db.add_vector('vec5', np.random.rand(5), metadata= { 'text': 'Fifth item' })
        
query_vector = np.random.rand(5)

print("Query vector", query_vector)

results = db.search(query_vector, top_k=3)
print(f"Query vector: {query_vector}\n")
for res in results:
    print(f"ID: {res['id']} | Similarity: {res['similarity']:.2f} | Metadata: {res['metadata']}")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-small-en-v1.5')

sentences = [
    "Kunal's cat is playing with wool.",
    "Ethan was born on 1st September 1700.",
    "Dogs are loyal animals.",
    "I love eating pizza but only on Fridays.",
    "Manika was born in Bhopal."
]

db = VectorDatabase()

for idx, sentence in enumerate(sentences):

    embedding = model.encode(sentence)

    db.add_vector(vec_id=f"sent_{idx}", vector=embedding, metadata={'sentence': sentence})

    query = "When is Ethan's birthday?"
    query_vec = model.encode(query)

    results = db.search(query_vec, top_k=3)
    print(f"Query: \"{query}\"\n")

    for res in results:
        print(f"Similar Sentence: {res['metadata']['sentence']}")
        print(f"Cosine Similarity Score: {res['similarity']:.2f}\n")

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
llm_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def generate_rag_response(query):

    context = ""
    query_embedding = embedder.encode(query, convert_to_numpy=True).astype('float32')

    results = db.search(query_embedding, top_k=3)

    for res in results:
        context += f"{res['metadata']['sentence']}\n"

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer accurately."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors='pt').to(llm_model.device)
    generated_ids = llm_model.generate(**model_inputs, max_new_tokens=512)

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

print(generate_rag_response("When is Ethan's birthday?"))

company_db = VectorDatabase()

# Hypothetical company information
company_information = [
    "Quantum Horizons Inc. is a pioneering space exploration company founded in 2030.",
    "The company specializes in developing quantum-powered spacecraft for interplanetary travel.",
    "With a team of 500 aerospace engineers and quantum physicists, Quantum Horizons is pushing the boundaries of space technology.",
    "Their flagship project, the 'StarLeap', aims to reduce travel time to Mars from months to just weeks.",
    "Quantum Horizons has established the first permanent research base on the Moon's far side.",
    "The company's innovative quantum propulsion system has revolutionized the concept of space travel.",
    "Headquartered in a state-of-the-art facility in Houston, Quantum Horizons also maintains orbital research stations.",
    "They've partnered with major space agencies worldwide to advance human presence in the solar system.",
    "Quantum Horizons' CEO, Dr. Zara Novak, is a former astronaut and a leading expert in quantum mechanics.",
    "The company's mission is to make interplanetary travel accessible and establish humanity as a multi-planet species."
]

companyModel = SentenceTransformer('all-MiniLM-L6-v2')

for idx, sentence in enumerate(company_information):
    embedding = model.encode(sentence)
    company_db.add_vector(vec_id=f"sentence_{idx}", vector=embedding,metadata={'sentence': sentence})

from crewai.tools import tool

@tool("RAG Tool")

def rag_tool(question: str) -> str:
    """Tool to search for relevant information from a vector database."""
    query_vec = companyModel.encode(question)

    results = company_db.search(query_vec, top_k=5)
    context = "\n".join([f"-{res['metadata']['sentence']}" for res in results])

    prompt = f"""You are a helpful assistant. Use the context below to answer the user's question.
            Context:
            {context}
            Question: {question}
            Answer:
            """

    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = llm_model.generate(**inputs, max_new_tokens=512, do_sample=False)
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

print(rag_tool.run("What is Quantum Horizons?"))


api_key = os.getenv("SERPER_API_KEY")

@tool("Web Search Tool")
def web_search_tool(query: str) -> str:
    """Tool to search the web for relevant information."""
    
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    
    payload = {
        "q": query
    }

    response = requests.post(url, headers = headers, json = payload)
    if response.status_code != 200:
        raise Exception(f"Request failed with status code: {response.status_code}")

    data = response.json()

    # Get value associated with key "organic" else return []
    search_results = data.get("organic", [])

    if not search_results:
        return "No search results found."
    
    context = ""

    for result in search_results:
        title = result.get("title", "")
        link = result.get("link", "")
        snippet = result.get("snippet", "")
        context += f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n\n"
    return f"Web Search Results:\n{context}"

print(web_search_tool.run("Important AI innovations of 2025"))

from crewai import Agent
from crewai import LLM

llm = LLM(
    model="ollama/qwen2.5:1.5b",
    base_url="http://localhost:11434"
)

# Agent 1: Retriever Agent
retriever_agent = Agent(
    role="Retriever Agent",
    goal="Retrieve the most relevant information to answer the user's query: {user_query}",
    backstory=(
         "You're a helpful agent. "
        "You're an expert at finding the right information to answer a user's query. "
        "You are great at following instructions and sequentially picking tools for information retrieval. "
        "You have decades of experience doing this."
    ),
    llm=llm,
    tools=[web_search_tool],
    verbose=True
)

# Agent 2: Customer Support Agent
customer_support_agent = Agent(
    role="Senior Customer Support Agent",
    goal=(
           "Accurately and concisely answer the user's query: {user_query} using the retrieved information. "
           "If you are unable to answer the query, apologise and tell that you do not have all the information you need to answer the query."
    ),
    backstory=(
         "You are a helpful senior customer support agent. "
         "You have decades of experience in answering user queries grounded to accurate information."
    ),
    llm=llm,
    verbose=True,
)

# Creating Tasks

from crewai import Task

# Task 1: Retriever Task
retrieval_task = Task(
    description=(
        "Retrieve the most relevant information from the given sources to answer the user's query: {user_query}. "
        "ALWAYS use the RAG Tool first. "
        "If you cannot find the required information, ONLY THEN use the Web Search Tool. "
        "DO NOT USE the Web Search Tool if you have sufficient information to accurately answer the user's query."
    ),
    expected_output="The most relevant information from the given sources to answer the user's query in a text format.",
    agent=retriever_agent,
)

# Task 2: Customer Support Task
customer_support_task = Task(
    description=(
        "Using the retrieved information, accurately and concisely answer the user's query: {user_query}."
    ),
    expected_output=(
        "Concise and accurate response based on the retrieved information given the user query: {user_query}. "
        "If you are unable to answer the query, apologise and inform the user that you do not have all the necessary information."
    ),
    agent=customer_support_agent,
    context=[retrieval_task],  # This task will use the output from the previous task as its context
)

# Create Crew
from crewai import Crew
from crewai.process import Process

customer_support_crew = Crew(
    agents = [retriever_agent, customer_support_agent],
    tasks = [retrieval_task, customer_support_task],
    verbose = True,
    process = Process.sequential
)

# Crew inputs

crew_inputs = {
    "user_query": "What is the name of the flagship project of the company?",
}

# Run the crew
result = customer_support_crew.kickoff(inputs = crew_inputs)
print(result.raw)

crew_inputs_2 = {
    "user_query": "Who is the winner of the 2024 Nobel prize in Physics?"
}

result_2 = customer_support_crew.kickoff(inputs = crew_inputs_2)
print(result_2.raw)