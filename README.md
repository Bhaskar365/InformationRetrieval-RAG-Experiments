# 🔍 Information Retrieval, RAG & Agentic AI

> A hands-on repository exploring **Information Retrieval (IR)**, **Retrieval-Augmented Generation (RAG)**, **Hybrid Retrieval**, and **Agentic AI** through practical implementations using **FAISS**, **Sentence Transformers**, **CrewAI**, and local LLMs.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)
![CrewAI](https://img.shields.io/badge/CrewAI-Agentic%20AI-purple)
![RAG](https://img.shields.io/badge/RAG-Implemented-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📖 Overview

This repository contains my implementations and experiments covering the complete retrieval pipeline—from **semantic search** to **Retrieval-Augmented Generation (RAG)** and **multi-agent AI systems**.

The goal of this repository is to understand and implement the core building blocks used in modern AI assistants, search engines, and enterprise knowledge systems.

---

## ✨ Implemented Features

* 🔎 Semantic Search using Sentence Transformers
* 📦 FAISS Vector Search
* 🌐 HNSW Indexing
* 📚 IVF Flat Indexing
* ⚖️ Reciprocal Rank Fusion (RRF)
* 🎯 Cross-Encoder Reranking
* 🧹 Duplicate Document Detection
* 📄 Company Knowledge Base Retrieval
* 🤖 Manual RAG Pipeline
* 🛠️ Custom RAG Tool
* 🌍 Web Search Tool Integration
* 👥 Agentic RAG using CrewAI
* 🔄 Multi-Agent Workflows
* 🦙 Local LLM Integration (Ollama + Qwen)

---

## 🛠️ Implemented Techniques

| Category                | Techniques                                            |
| ----------------------- | ----------------------------------------------------- |
| **Embeddings**          | Sentence Transformers                                 |
| **Vector Search**       | FAISS                                                 |
| **Approximate Search**  | HNSW                                                  |
| **Scalable Indexing**   | IVF Flat                                              |
| **Ranking**             | Reciprocal Rank Fusion (RRF), Cross-Encoder Reranking |
| **Retrieval**           | Semantic Search, Company Guideline Retrieval          |
| **Document Processing** | Duplicate Detection                                   |
| **RAG**                 | Manual Retrieval-Augmented Generation                 |
| **Agentic AI**          | CrewAI Multi-Agent Workflow                           |
| **Tools**               | RAG Tool, Web Search Tool                             |
| **LLMs**                | Hugging Face Transformers, Ollama                     |

---

## 🤖 Agentic RAG Workflow

```text
                 User Query
                      │
                      ▼
              Retriever Agent
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
 Company RAG Tool          Web Search Tool
         │                         │
         └────────────┬────────────┘
                      ▼
             Retrieved Context
                      │
                      ▼
       Customer Support Agent
                      │
                      ▼
              Grounded Response
```

---

## 💻 Tech Stack

* Python
* FAISS
* Sentence Transformers
* Hugging Face Transformers
* CrewAI
* Ollama
* NumPy
* Pandas
* scikit-learn
* Requests

---

## 🎯 Learning Goals

This repository explores concepts including:

* Dense Vector Retrieval
* Approximate Nearest Neighbor Search
* Retrieval-Augmented Generation (RAG)
* Hybrid Retrieval
* Ranking & Reranking
* Multi-Agent Systems
* Tool Calling
* Local LLM Integration
* AI Workflow Orchestration

---

## 🚀 Future Improvements

* BM25 Retrieval
* Hybrid Search (BM25 + Dense Retrieval)
* Metadata Filtering
* Query Rewriting
* Context Compression
* Retrieval Evaluation Benchmarks
* LangSmith Tracing & Evaluation
* Agent Memory
* Structured Outputs
* FastAPI Deployment
* Docker Support
* Vector Databases (Milvus, Qdrant, ChromaDB)

---

## 📸 Repository Highlights

The repository contains implementations and execution examples for:

* Semantic Search
* FAISS Indexing
* HNSW & IVF Flat
* Reciprocal Rank Fusion (RRF)
* Cross-Encoder Reranking
* Duplicate Detection
* Manual RAG
* Company Knowledge Base Retrieval
* CrewAI Agent Workflows
* Agentic RAG
* Local LLM Tool Calling

---

## 📌 Disclaimer

This repository is intended for **learning, experimentation, and research** in **Information Retrieval**, **Retrieval-Augmented Generation**, and **Agentic AI**. The focus is on understanding the underlying concepts and implementing them from first principles before moving toward production-grade systems.

---

## 📖 Required things for running this program

* CrewAI package
* Free Serper API key from _https://serper.dev/_ with 2500 free credit
* Ollama with Qwen2.5:1.5B model running with localhost _11434_ to call Agent LLM instead of default OpenAI API. Skip if already have key. Hugging Face Text Generation Inference (TGI) API, vLLM or LM Studio also optional 
* Necessary packages for running RAG program and Python 3.10 / 3.12. Python 3.14 not compatible for CrewAI.  

---

## 📖 Screenshot of Outputs obtained from CrewAI Agents. Note, not all screenshots were included to make the screenshots limited, few outputs were skipped
<br>


<img width="1403" height="747" alt="image" src="https://github.com/user-attachments/assets/92d3cae2-2b98-40ed-a147-cb70281a5ad6" />

<img width="1391" height="751" alt="image" src="https://github.com/user-attachments/assets/eb653452-0b9a-4a92-9741-20d897c6851b" />

<img width="1383" height="745" alt="image" src="https://github.com/user-attachments/assets/578cef66-0d77-4c29-9d85-58277e25283d" />

<img width="1387" height="775" alt="image" src="https://github.com/user-attachments/assets/eda99c15-3dd2-44f3-b243-dca30f0ff989" />

<img width="1391" height="771" alt="image" src="https://github.com/user-attachments/assets/eaa424c0-c838-48e6-9804-09dce9b26ad6" />

<img width="1383" height="767" alt="image" src="https://github.com/user-attachments/assets/ea45fd2a-9eb3-4732-9352-17730db96321" />

<img width="1389" height="677" alt="image" src="https://github.com/user-attachments/assets/b2573a6d-7fba-4965-8597-0eec082b5728" />

<img width="1387" height="757" alt="image" src="https://github.com/user-attachments/assets/ff9c109a-ae19-4efa-a78b-35cea04e8084" />

<img width="1383" height="705" alt="image" src="https://github.com/user-attachments/assets/fffbcecc-0faf-463e-9db4-2f87004706db" />
















