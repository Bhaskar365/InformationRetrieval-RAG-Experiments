# 🔍 Information Retrieval, RAG & Agentic AI

> A hands-on repository exploring **Information Retrieval (IR)**, **Retrieval-Augmented Generation (RAG)**, **Hybrid Retrieval**, **Reranking**, **Conversational RAG**, and **Agentic AI** through practical implementations using **FAISS**, **Sentence Transformers**, **LangChain**, **CrewAI**, and local LLMs.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)
![CrewAI](https://img.shields.io/badge/CrewAI-Agentic%20AI-purple)
![RAG](https://img.shields.io/badge/RAG-Implemented-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📖 Overview

This repository contains practical implementations and experiments covering the retrieval and generation pipeline—from **semantic search and approximate nearest-neighbor indexing** to **RAG, conversational RAG, reranking, citation-aware generation, and multi-agent AI systems**.

The project focuses on understanding how modern AI assistants and enterprise knowledge systems are built, starting from retrieval fundamentals and gradually moving toward more production-oriented RAG architectures.

The implementations intentionally cover both **from-scratch IR concepts** and higher-level frameworks such as **LangChain** and **CrewAI**.

---

## ✨ Implemented Features

### 🔎 Information Retrieval

* Semantic Search using Sentence Transformers
* FAISS Vector Search
* HNSW Approximate Nearest Neighbor Indexing
* IVF Flat Indexing
* Reciprocal Rank Fusion (RRF)
* Cross-Encoder Reranking
* MMR (Maximal Marginal Relevance) Retrieval
* Duplicate Document Detection
* Company Knowledge Base Retrieval

### 📚 RAG

* Manual RAG Pipeline
* LangChain RAG Pipeline
* History-Aware Retrieval
* Conversational RAG
* Contextual Compression Retrieval
* Cross-Encoder based document reranking
* Citation-aware document processing
* Local LLM-based RAG using Ollama
* Streaming RAG responses
* Session-based conversation history
* `Runnable`-based LangChain architecture

### 🛠️ RAG & Search Tools

* Custom RAG Tool
* Web Search Tool Integration
* Retrieval-based context generation
* Reranking retrieved documents before generation

### 🤖 Agentic AI

* CrewAI Multi-Agent Workflow
* Agentic RAG
* Retriever Agent
* Customer Support Agent
* RAG Tool integration
* Web Search Tool integration
* Multi-Agent Workflows
* Local LLM integration for agents
* Tool-based agent workflows

---

## 🏗️ Current RAG Architecture

The current conversational RAG implementation follows a pipeline similar to:

```text
                         User Query
                              │
                              ▼
                  RunnableWithMessageHistory
                              │
                              ▼
                  History-Aware Retriever
                              │
                              ▼
                    Vector Retrieval
                              │
                       MMR Retrieval
                     k = 20 / fetch_k = 40
                              │
                              ▼
                  Contextual Compression
                              │
                              ▼
                  Cross-Encoder Reranker
                   BAAI/bge-reranker-large
                              │
                              ▼
                         Top 5 Docs
                              │
                              ▼
                    Citation Assignment
                       [1], [2], [3]...
                              │
                              ▼
                     QA / Prompt Chain
                              │
                              ▼
                         Ollama LLM
                              │
                              ▼
                    Streaming Answer
                              │
                              ▼
                 Citation + Source Mapping
```

The current retrieval architecture intentionally separates **initial document retrieval** from **reranking**, allowing a larger candidate set to be retrieved before selecting the most relevant documents for generation.

---

## 🧠 Retrieval Strategy

The current RAG retriever uses a two-stage retrieval strategy.

### Stage 1 — Candidate Retrieval

FAISS/vector retrieval is combined with **MMR (Maximal Marginal Relevance)** to retrieve a diverse set of candidate documents.

Current configuration:

```text
Initial candidates:
k = 20

MMR candidate pool:
fetch_k = 40
```

### Stage 2 — Reranking

The retrieved candidates are passed through a cross-encoder reranker:

```text
Model:
BAAI/bge-reranker-large

Final documents:
top_n = 5
```

Conceptually:

```text
Query
  │
  ▼
Vector Search
  │
  ▼
MMR
  │
  ▼
20 candidate documents
  │
  ▼
Cross-Encoder Reranker
  │
  ▼
Top 5 relevant documents
  │
  ▼
LLM
```

This approach allows the system to retrieve a relatively broad candidate set while using the more expensive cross-encoder model for final relevance ranking.

---

## 💬 Conversational RAG

The RAG pipeline supports multi-turn conversations using LangChain's `RunnableWithMessageHistory`.

For example:

```text
User:
What is the company's revenue?

Assistant:
The company reported revenue of $10 million.

User:
What about the previous year?

Assistant:
The previous year's revenue was $8 million.
```

The system uses a **history-aware retriever** to transform follow-up questions into standalone queries before retrieval.

Conceptually:

```text
Chat History
     │
     ▼
Question Rewriting
     │
     ▼
Standalone Question
     │
     ▼
Retriever
```

This allows follow-up questions such as:

```text
"What about last year?"
```

to be interpreted using the previous conversation.

---

## 📑 Citation-Aware RAG

The current RAG architecture is being extended to support source citations.

Retrieved documents are assigned citation identifiers:

```text
[1]
[2]
[3]
```

The LLM is instructed to reference these identifiers when generating factual claims.

Example:

```text
The company reported revenue of $10 million in 2025 [1].
```

The citation mapping is maintained separately from the generated text:

```text
[1] company_report.pdf - Page 12
[2] financial_report.pdf - Page 8
```

An important design principle is that citations should correspond to the **same documents used by the generation pipeline** rather than performing a second independent retrieval solely to generate citations.

---

## 🌊 Streaming RAG

The conversational RAG pipeline supports streaming responses using LangChain Runnable streaming.

Instead of waiting for the complete LLM response:

```text
The company reported revenue...
```

the application can receive generated content incrementally:

```text
The
company
reported
revenue
...
```

This allows the same RAG backend to eventually support:

* CLI streaming
* Web applications
* FastAPI streaming endpoints
* Server-Sent Events (SSE)
* Interactive chat interfaces

---

## 🤖 Agentic RAG Workflow

The repository also contains an agentic RAG implementation using CrewAI.

```text
                 User Query
                      │
                      ▼
              Retriever Agent
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   Company RAG Tool        Web Search Tool
          │                       │
          └───────────┬───────────┘
                      ▼
              Retrieved Context
                      │
                      ▼
         Customer Support Agent
                      │
                      ▼
              Grounded Response
```

The agentic workflow explores how retrieval and external tools can be combined with LLM-based reasoning.

---

## 🛠️ Implemented Techniques

| Category                | Implemented Techniques                   |
| ----------------------- | ---------------------------------------- |
| **Embeddings**          | Sentence Transformers                    |
| **Vector Search**       | FAISS                                    |
| **Approximate Search**  | HNSW, IVF Flat                           |
| **Retrieval**           | Semantic Search, MMR                     |
| **Ranking**             | RRF, Cross-Encoder Reranking             |
| **Compression**         | Contextual Compression                   |
| **RAG**                 | Manual RAG, LangChain RAG                |
| **Conversational RAG**  | History-Aware Retrieval, Message History |
| **Streaming**           | LangChain Runnable Streaming             |
| **Citations**           | Citation-aware document processing       |
| **Document Processing** | Duplicate Detection                      |
| **Tools**               | RAG Tool, Web Search Tool                |
| **Agentic AI**          | CrewAI Multi-Agent Workflow              |
| **LLMs**                | Hugging Face Transformers, Ollama        |
| **Local AI**            | Ollama-based local inference             |

---

## 🛠️ Current Tech Stack

* Python
* LangChain
* LangChain Runnables
* FAISS
* Sentence Transformers
* Hugging Face Transformers
* Cross-Encoder Models
* CrewAI
* Ollama
* NumPy
* Pandas
* scikit-learn

---

## 🎯 Learning Goals

This repository explores the complete progression from classical information retrieval concepts to modern RAG and agentic architectures.

### Information Retrieval

* Dense Vector Retrieval
* Semantic Search
* Approximate Nearest Neighbor Search
* HNSW
* IVF
* MMR
* Reciprocal Rank Fusion
* Cross-Encoder Reranking

### Retrieval-Augmented Generation

* Document ingestion
* Chunking
* Embeddings
* Vector search
* Context construction
* Reranking
* Prompt design
* Conversational RAG
* Citation-aware generation
* Streaming generation

### Agentic AI

* Tool calling
* RAG tools
* Web search tools
* Multi-agent workflows
* Agent coordination
* Local LLM agents

---

# 🚧 Future Scope

The next phase of the project focuses on moving from **working RAG implementations** toward **measurable and production-oriented RAG systems**.

### 🔬 Retrieval Evaluation

* Retrieval Recall@K
* Precision@K
* Mean Reciprocal Rank (MRR)
* Reranker evaluation
* Retrieval latency benchmarking
* RAG evaluation datasets
* Regression testing for retrieval quality

### 🔍 Retrieval Improvements

* BM25 retrieval
* Hybrid BM25 + dense retrieval
* Query expansion
* Query decomposition
* Improved metadata filtering
* Retrieval confidence scoring
* Adaptive retrieval strategies

### 📑 Citation Reliability

* Citation correctness evaluation
* Claim-to-document verification
* Citation grounding
* Source confidence scoring
* Clickable document/page citations

### 📊 Observability & Evaluation

* LangSmith tracing
* RAG evaluation pipelines
* Prompt/version tracking
* Retrieval and generation tracing
* Latency monitoring
* Token usage monitoring
* Failure analysis

### 🧠 Conversation & Memory

* Persistent conversation storage
* Redis/PostgreSQL-backed message history
* Conversation summarization
* Long-context memory management
* Session management

### ⚙️ Production Engineering

* Structured logging
* Error handling
* Retries and exponential backoff
* Timeouts
* Caching
* Configuration management
* Automated testing
* Load testing

### 🌐 Application & Deployment

* FastAPI backend
* Streaming APIs / Server-Sent Events
* Interactive RAG UI
* Docker
* Production deployment
* Health checks
* Monitoring

### 🔐 Security

* Authentication
* Authorization
* Document-level access control
* Multi-user / multi-tenant retrieval
* Prompt injection protection
* Secure document ingestion

### 🗄️ Data & Infrastructure

* Metadata filtering
* Document versioning
* Persistent vector databases
* Qdrant
* Milvus
* ChromaDB
* Production document ingestion pipelines

### 🤖 Advanced Agentic Systems

* Agent memory
* Structured outputs
* Tool routing
* Multi-step research workflows
* Agent evaluation
* Agent observability
* Human-in-the-loop workflows

---

## Current Overall Architecture

                         ┌─────────────────┐
                         │     Documents   │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │     Loaders     │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Metadata + IDs  │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    Chunking     │
                         │ 1500 / 200      │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    chunk_id     │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
             ┌──────────────┐            ┌──────────────┐
             │    Chroma    │            │     BM25     │
             │ Dense Search │            │ Sparse Search│
             └──────┬───────┘            └──────┬───────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                       ┌─────────────────────┐
                       │  Ensemble Retriever │
                       │ BM25 40% / Vec 60%  │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  BGE Cross-Encoder  │
                       │      Reranker       │
                       └──────────┬──────────┘
                                  │
                                  ▼
                             ┌─────────┐
                             │ Top 5   │
                             │ Chunks  │
                             └────┬────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
             ┌──────────────┐           ┌──────────────┐
             │   Context    │           │  Citations   │
             └──────┬───────┘           └──────┬───────┘
                    │                           │
                    ▼                           │
             ┌──────────────┐                   │
             │ History-aware│                   │
             │  Retriever   │                   │
             └──────┬───────┘                   │
                    │                           │
                    ▼                           │
             ┌──────────────┐                   │
             │     LLM      │                   │
             └──────┬───────┘                   │
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                         ┌─────────────────┐
                         │ Answer + Sources│
                         └─────────────────┘

## 📈 Production RAG Roadmap

The project is progressing through the following stages:

```text
                    IR FUNDAMENTALS
                          │
                          ▼
                  Semantic Search
                          │
                          ▼
                    FAISS / ANN
                          │
                          ▼
               MMR + Reranking
                          │
                          ▼
                   Basic RAG
                          │
                          ▼
              Conversational RAG
                          │
                          ▼
             Streaming + Citations
                          │
                          ▼
               ┌──────────────────┐
               │ Evaluation       │
               │ Recall@K / MRR   │
               └────────┬─────────┘
                        ▼
               Hybrid Retrieval
                        │
                        ▼
              Citation Verification
                        │
                        ▼
                 Observability
                        │
                        ▼
               Persistent Memory
                        │
                        ▼
               API + Deployment
                        │
                        ▼
              Security + Scaling
                        │
                        ▼
             Production RAG System
```

---

## 📸 Repository Highlights

The repository contains implementations and execution examples for:

* Semantic Search
* FAISS Vector Search
* HNSW Indexing
* IVF Flat Indexing
* Reciprocal Rank Fusion
* Cross-Encoder Reranking
* MMR Retrieval
* Duplicate Detection
* Company Knowledge Base Retrieval
* Manual RAG
* LangChain RAG
* Conversational RAG
* Streaming RAG
* Citation-Aware RAG
* CrewAI Agent Workflows
* Agentic RAG
* RAG and Web Search Tools
* Local LLM Tool Calling

Screenshots and execution examples are included for selected experiments and agent workflows.

---

## 📌 Disclaimer

This repository is intended for **learning, experimentation, research, and progressive system development** in **Information Retrieval, Retrieval-Augmented Generation, and Agentic AI**.

The project emphasizes understanding the underlying concepts by implementing retrieval, ranking, RAG, and agentic workflows before progressively introducing production-oriented engineering practices.

The implementations are experimental and may evolve as retrieval evaluation, observability, deployment, and production engineering capabilities are added.

---

## 📖 Requirements

### Python

Recommended Python versions:

```text
Python 3.10
Python 3.12
```

Python 3.14 may not currently be compatible with some CrewAI-related dependencies used by the repository.

### Local LLM

Ollama can be used for local LLM inference.

Example:

```bash
ollama run qwen2.5:1.5b
```

The repository also experiments with other local inference approaches such as Hugging Face-based inference and can be extended to work with:

* TGI
* vLLM
* LM Studio

### Agentic AI

CrewAI is required for the agentic workflow implementations.

### Web Search

The agentic web-search implementation requires a Serper API key.

A free Serper account can be used for experimentation.

### RAG Dependencies

Depending on the implementation being executed, additional packages may be required for:

* FAISS
* Sentence Transformers
* Hugging Face Transformers
* LangChain
* Ollama integration
* Cross-Encoder reranking
* Vector databases

Refer to the project's dependency files for the exact installation requirements.

---

## 🚀 Project Direction

The project is evolving from:

```text
Learning IR concepts
        ↓
Implementing retrieval algorithms
        ↓
Building RAG pipelines
        ↓
Adding conversational capabilities
        ↓
Adding reranking and citations
        ↓
Building agentic workflows
        ↓
Evaluating retrieval quality
        ↓
Adding observability
        ↓
Hardening reliability and security
        ↓
Production-oriented RAG
```

The long-term goal is to understand not only **how to build RAG and agentic systems**, but also **how to evaluate, debug, optimize, secure, and deploy them as reliable AI applications**.

---
📸 **Screenshots**

This is agentic RAG from CrewAI screenshots. Other screenshots are not attached; only representative outputs are included. The final production application screenshots will be provided once the complete application is finalized.

---
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
















