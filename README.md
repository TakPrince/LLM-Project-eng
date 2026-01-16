
# 📘 `README.md`

# 🤖 Autonomous AI Deal Intelligence System

An end-to-end **Agentic AI system** that automatically finds online deals, estimates the true value of products using neural networks and RAG, and notifies users about the best bargains — without human intervention.

This project is built as a progressive AI engineering journey (Day 1 → Day 8), evolving from basic scraping to a fully autonomous multi-agent AI system.

---

## 🚀 Project Overview

This system works like an intelligent deal hunter:

1. Scrapes real-world product deals from the internet  
2. Stores product data in a vector database (ChromaDB)  
3. Uses Retrieval-Augmented Generation (RAG) for similarity search  
4. Uses neural networks to predict real product value  
5. Uses LLMs for reasoning and decision-making  
6. Uses a multi-agent architecture for orchestration  
7. Automatically notifies users of the best bargains  

---

## 🧠 System Architecture


```
ScannerAgent ───► Finds online deals
│
▼
FrontierAgent ───► RAG-based price estimation (LLM)
NeuralAgent   ───► Neural network price prediction
SpecialistAgent ─► External compute (Modal)
│
▼
EnsembleAgent ───► Aggregates all estimates
│
▼
AutonomousPlanningAgent ─► Selects best bargain
│
▼
MessagingAgent ───► Sends notification to user
```

## 📂 Project Structure

```
.vscode/
├── day1/   → Web scraping + LLM intro
├── day2/   → Chatbots, APIs, memory, Gradio
├── day4/   → System inspection & code generation
├── day5/   → RAG system with ChromaDB
├── day6/   → Neural pricing model training
├── day7/   → Evaluation + ensemble logic
├── day8/   → Full multi-agent autonomous AI system
```



## 🔹 Day-wise Highlights

### ✅ Day 1 — Web Scraping + LLM
- Scraper engine
- Brochure generation using LLMs

### ✅ Day 2 — Chatbots & APIs
- Multi-API chatbot
- Memory-based assistant
- Gradio UI
- SQLite storage

### ✅ Day 4 — Code Intelligence
- System inspection
- Code generation

### ✅ Day 5 — RAG System
- ChromaDB vector store
- Embedding pipelines
- Retrieval-based QA system
- Evaluation framework

### ✅ Day 6 — Neural Pricing Model
- Dataset creation (JSONL)
- Deep neural network for price prediction
- Training & inference pipeline

### ✅ Day 7 — Model Evaluation
- Pricing evaluation
- Ensemble modeling

### ✅ Day 8 — Autonomous Agent System
- Tool calling
- Planning loops
- Memory
- Multi-agent orchestration
- Notification system

---

## 🛠 Tech Stack

- Python
- Open-source LLMs (Groq / OpenRouter / LiteLLM)
- ChromaDB (Vector Database)
- SentenceTransformers
- PyTorch (Neural Networks)
- Agentic Architecture
- Tool Calling
- RAG (Retrieval-Augmented Generation)

---

## 🎯 Key Features

- Autonomous decision-making
- Multi-agent collaboration
- Neural + LLM hybrid intelligence
- Vector search with embeddings
- Real-world data scraping
- User notification system

---

## 🧾 Use Case

> Automatically detect underpriced products on the internet and notify users with the best deals.

---

---

## 🤖 Agent Responsibilities

| Agent | Responsibility |
|------|----------------|
| ScannerAgent | Scrapes live deals from RSS feeds |
| FrontierAgent | Uses RAG + LLM to estimate price |
| NeuralNetworkAgent | Predicts price using deep learning |
| SpecialistAgent | External GPU inference via Modal |
| EnsembleAgent | Combines all predictions |
| AutonomousPlanningAgent | Orchestrates all tools |
| MessagingAgent | Sends user notifications |

---

## ⚙️ Core Intelligence Stack

- **LLMs (Open-source)** — Reasoning & planning
- **Neural Networks (PyTorch)** — Price prediction
- **RAG (ChromaDB)** — Similar product search
- **Vector Embeddings** — Semantic retrieval
- **Tool Calling** — Agent orchestration
- **Memory System** — Persistent deal tracking

---


## 👨‍💻 Author

Prince Tak  
MSc Computer Science (AI/ML) — MIT ADT University  
BSc IT — Indus University  

---

## ⭐ Final Note

This project demonstrates real-world AI engineering practices used in modern production AI systems:
- Agentic AI
- Decision intelligence
- RAG pipelines
- Neural prediction systems
- Autonomous workflows
```
