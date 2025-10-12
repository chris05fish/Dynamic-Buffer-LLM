# save as dynamic_buffer_demo.py
import os, time, json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI  # new SDK
from dotenv import load_dotenv

load_dotenv()  # load .env file

# ========== CONFIG ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

EMBED_MODEL = "text-embedding-3-small"  # change if needed
GPT_MODEL = "gpt-4o-mini"               # or gpt-4o, or gpt-4, etc.

# If you prefer local embeddings, uncomment:
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers (fast)
use_local_embeddings = True

# buffer params
BASE_TOP_K = 6             
INITIAL_BUFFER_K = 4       
EXPAND_FACTOR = 2          
MAX_BUFFER_TOKENS = 2500   
# ============================

# ---------- Embedding helpers ----------
if use_local_embeddings:
    sbert = SentenceTransformer(LOCAL_EMBED_MODEL)

def embed_texts(texts):
    if use_local_embeddings:
        return sbert.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # else use OpenAI embeddings
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([r.embedding for r in resp.data], dtype=np.float32)

# ---------- Simple in-memory vector store using FAISS ----------
class SimpleVectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.ids = []
        self.metadatas = []
        self.texts = []
        self.index = faiss.IndexFlatL2(dim)
    def add(self, vectors, texts, metadatas):
        self.index.add(np.array(vectors).astype(np.float32))
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self.ids.extend(list(range(len(self.ids), len(self.ids)+len(texts))))
    def search(self, vector, k=5):
        D, I = self.index.search(np.array([vector]).astype(np.float32), k)
        results = []
        for idx in I[0]:
            if idx < len(self.texts):
                results.append({"text": self.texts[idx], "meta": self.metadatas[idx]})
        return results

# ---------- Ingest pipeline ----------
def ingest_conversation(messages, store):
    chunks, metas = [], []
    for m in messages:
        text = m["text"]
        parts = [text] if len(text.split()) < 120 else text.split(". ")
        for p in parts:
            chunks.append(p.strip())
            metas.append({"speaker": m["speaker"], "ts": m["ts"], "orig_id": m["id"]})
    vecs = embed_texts(chunks)
    store.add(vecs, chunks, metas)

# ---------- Context Manager: dynamic buffer ----------
def build_dynamic_buffer(query, store, current_buffer_ids=None):
    q_emb = embed_texts([query])[0]
    top = store.search(q_emb, k=BASE_TOP_K*EXPAND_FACTOR)
    now = time.time()
    scored = []
    for r in top:
        txt = r["text"]
        meta = r["meta"]
        vec = embed_texts([txt])[0]
        sim = np.dot(q_emb, vec) / (np.linalg.norm(q_emb) * np.linalg.norm(vec) + 1e-9)
        recency = np.exp(-(now - meta["ts"]) / (60*60*24*30))  # 30-day decay
        score = 0.7*sim + 0.3*recency
        scored.append((score, txt, meta))
    scored = sorted(scored, key=lambda x: -x[0])
    buffer = scored[:INITIAL_BUFFER_K]
    avg_sim = sum([s for s,_,_ in buffer]) / len(buffer)
    if avg_sim < 0.4:
        k = min(len(scored), INITIAL_BUFFER_K * EXPAND_FACTOR)
        buffer = scored[:k]
    return buffer

# ---------- Prompt assembly ----------
def assemble_prompt(buffer, user_query, system_prompt="You are a helpful assistant. Use the context below when answering."):
    context_blocks = []
    for score, txt, meta in buffer:
        context_blocks.append(f"[{meta['speaker']} @ {time.ctime(meta['ts'])}]\n{txt}\n---\n")
    context_text = "\n".join(context_blocks)
    prompt = f"{system_prompt}\n\nCONTEXT:\n{context_text}\n\nUSER: {user_query}\n\nAnswer concisely, and say which context items you used."
    return prompt

# ---------- Call LLM (updated for OpenAI Python >=1.0) ----------
def call_gpt(prompt):
    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":prompt}
        ],
        max_tokens=400
    )
    return resp.choices[0].message.content

# ========== Example usage ==========
if __name__ == "__main__":
    now = time.time()
    messages = [
        {"id":1,"speaker":"User","text":"I started a project called Live App to help people find friends at events.","ts": now - 60*60*24*10},
        {"id":2,"speaker":"User","text":"We discussed improving the UI for check-in flow; Alex suggested a spinner and PIN.","ts": now - 60*60*24*9},
        {"id":3,"speaker":"User","text":"I uploaded a design doc about UX and a link to prototypes.","ts": now - 60*60*24*2},
        {"id":4,"speaker":"User","text":"I prefer minimalist design and fast onboarding. Also I hate popups.","ts": now - 60*60*24*1},
    ]

    dim = 384 if use_local_embeddings else 1536
    store = SimpleVectorStore(dim)
    ingest_conversation(messages, store)

    user_query = "What did Alex suggest for the check-in flow?"
    buffer = build_dynamic_buffer(user_query, store)
    prompt = assemble_prompt(buffer, user_query)
    print("PROMPT:\n", prompt[:1000], "...\n")

    # Call GPT
    answer = call_gpt(prompt)
    print("GPT ANSWER:\n", answer)
