# 🔍 Information Retrieval & RAG Experiments

> A research and implementation repository exploring modern **Information Retrieval (IR)** and **Retrieval-Augmented Generation (RAG)** techniques using **FAISS** and related algorithms.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)
![RAG](https://img.shields.io/badge/RAG-Experiments-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📖 Overview

This repository contains implementations and experiments with a variety of retrieval techniques used in **semantic search** and **Retrieval-Augmented Generation (RAG)** systems.

Originally developed as a **FAISS semantic search prototype**, the project has evolved into a collection of retrieval, indexing, ranking, and document processing techniques for building scalable search and RAG pipelines.

---

## ✨ Features

- 🔎 Semantic search using sentence embeddings
- 📦 FAISS vector indexing
- 🌐 HNSW (Hierarchical Navigable Small World) indexing
- 📚 IVF Flat indexing
- ⚖️ Reciprocal Rank Fusion (RRF), Cross Encoder
- 🧹 Text duplicate detection
- 🤖 Manual RAG pipeline implementation
- 📄 Company guideline retrieval & document matching
- 📊 Retrieval evaluation experiments

---

## 🛠️ Implemented Techniques

| Technique | Purpose |
|-----------|---------|
| **FAISS** | Dense vector similarity search |
| **HNSW** | Approximate nearest neighbor search |
| **IVF Flat** | Scalable vector indexing |
| **Reciprocal Rank Fusion (RRF)** | Hybrid ranking and result fusion |
| **Duplicate Detection** | Identify and remove redundant documents |
| **Manual RAG** | Build a Retrieval-Augmented Generation pipeline from scratch |
| **Guideline Retrieval** | Match company documentation with user queries |

---

## 💻 Tech Stack

- Python
- FAISS
- Sentence Transformers
- NumPy
- Pandas
- scikit-learn

---

## 🎯 Project Goals

- Explore modern retrieval algorithms
- Compare different vector indexing strategies
- Experiment with ranking and fusion methods
- Build custom RAG pipelines without relying on external frameworks
- Investigate document preprocessing and duplicate detection techniques

---

## 🚀 Future Improvements

- BM25 integration
- Hybrid keyword + semantic retrieval
- Cross-encoder reranking
- Retrieval evaluation benchmarks
- LangChain / LlamaIndex integration
- Metadata-aware retrieval
- Vector database integration (Milvus, Qdrant, Chroma)
- Query expansion and relevance feedback

---

## 📌 Disclaimer

This repository is intended for **learning, experimentation, and research** in Information Retrieval and Retrieval-Augmented Generation. It is not designed as a production-ready search framework.

---
