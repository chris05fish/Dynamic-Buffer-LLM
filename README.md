# Dynamic Buffer LLM: Context-Aware Conversational Augmentation

![Dynamic Buffer Concept](https://img.shields.io/badge/status-experimental-orange) ![Python](https://img.shields.io/badge/python-3.11-blue) ![OpenAI](https://img.shields.io/badge/OpenAI-GPT4o--mini-green)

---

## 🚀 Project Overview

**Dynamic Buffer LLM** is a research-level Python prototype that enhances large language model (LLM) conversations by maintaining a **context-aware, dynamically resizing memory buffer**. This system bridges the gap between static retrieval-augmented generation (RAG) methods and truly adaptive, memory-informed dialogue management.

Unlike traditional RAG pipelines, this project:

- Dynamically selects the most relevant conversation chunks based on **semantic similarity and recency**.
- Expands or contracts the context buffer according to query difficulty and relevance.
- Uses embeddings (OpenAI or local SentenceTransformers) with **FAISS-based vector search**.
- Assembles LLM prompts enriched with context to generate **high-fidelity, contextually grounded responses**.

This is a step toward **“supermemory” LLMs**, enabling smarter and more coherent AI assistants for multi-turn, long-term conversations.

---

## 🧠 Key Features

1. **Dynamic Buffering**
   - Maintains a flexible window of relevant past conversation items.
   - Adjusts size and content based on similarity and recency scoring.
   - Reduces context bloat while improving LLM reasoning.

2. **Hybrid Embedding Support**
   - Local embeddings with SentenceTransformers (`all-MiniLM-L6-v2`) for offline use.
   - OpenAI embeddings (`text-embedding-3-small`) for cloud-based semantic search.

3. **In-Memory Vector Store**
   - FAISS index for efficient similarity search.
   - Tracks metadata such as speaker, timestamp, and chunk origin.

4. **LLM Integration**
   - Fully compatible with **OpenAI GPT-4o-mini** (or other GPT-4/GPT-3.5 models).
   - Assembles structured prompts for **retrieval-augmented, context-aware responses**.

---

## ⚡ Example

```python
# sample user query
user_query = "What did Alex suggest for the check-in flow?"

# build dynamic buffer
buffer = build_dynamic_buffer(user_query, store)

# assemble prompt
prompt = assemble_prompt(buffer, user_query)

# call GPT
answer = call_gpt(prompt)
print("GPT ANSWER:\n", answer)

Architect 
[Conversation History] --> [Chunking & Embedding] --> [FAISS Vector Store] 
      --> [Dynamic Buffer Selection] --> [Prompt Assembly] --> [LLM Response]

Installation:
git clone https://github.com/yourusername/dynamic-buffer-llm.git
cd dynamic-buffer-llm
pip install openai faiss-cpu sentence_transformers numpy scikit-learn python-dotenv   

Use dot env file for open ai api key

