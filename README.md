# **Meowtron — Offline AI Cat Assistant**

Meowtron is a lightweight offline AI system with a sarcastic, cat-like personality.
It detects emotions, retrieves memories, and generates personality-based responses using multiple small models.

---

## **🧠 System Flow**

Below is the simple flow of how Meowtron processes any user message:

```
 ┌──────────────────────┐
 │      User Input       │
 └──────────┬───────────┘
            ↓
 ┌──────────────────────┐
 │ Emotion Detection     │
 │  (DistilBERT)         │
 └──────────┬───────────┘
            ↓
 ┌────────────────────────────┐
 │ Query Processing            │
 │ (LLaMA 3.2 — 1B)            │
 └──────────┬─────────────────┘
            ↓
 ┌─────────────────────────────────┐
 │ Semantic Memory Retrieval       │
 │  • MiniLM-L6-v2 embeddings      │
 │  • FAISS vector search          │
 │  • SQLite memory database       │
 └──────────┬──────────────────────┘
            ↓
 ┌──────────────────────────────┐
 │ Prompt Builder                │
 │ (personality + emotion + mem)│
 └──────────┬───────────────────┘
            ↓
 ┌──────────────────────────────┐
 │ Main Response Generation      │
 │     (LLaMA 3.1)               │
 └──────────┬───────────────────┘
            ↓
 ┌──────────────────────────────┐
 │ Fact Extraction               │
 │  → FAISS + SQLite storage     │
 └──────────────────────────────┘
```

---

## **📁 Project Structure**

```
core/
 ├─ dataProcessing.py     # Query builder for memory search
 ├─ emotion.py            # DistilBERT emotion detection
 ├─ llm.py                # LLaMA wrappers
 ├─ memory.py             # FAISS + SQLite memory engine
 ├─ promtBuilder.py       # Final prompt generator
 └─ memoryOld.py          # Legacy system (unused)

models/                   # LLaMA model folders
all-MiniLM-L6-v2/         # Embedding model
distilbert_emotion_model/ # Emotion model
distilbert_emotion_tokenizer/
assets/                   # FAISS index + SQLite DB
main.py                   # Main entry point
memory.json               # Optional JSON backup memory
```

---

## **✨ Features**

* Offline LLM pipeline
* Emotion-aware responses
* Memory storage & semantic search
* Cat-like personality system
* Multi-model modular architecture

*(Still no GUI or voice because someone is procrastinating)*

---


