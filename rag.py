"""Step 2: the RAG logic (Retrieval-Augmented Generation).

    question -> turn into numbers (embedding) -> find similar chunks in ChromaDB
             -> give those chunks to an LLM (Groq) -> get a final answer

Importing this file loads the embedding model and fills the database once.
"""
import os

import chromadb
from sentence_transformers import SentenceTransformer

from chunking import load_all_chunks

# --- Settings you can change -------------------------------------------------
MIN_SCORE = 0.30   # below this similarity we say "not a policy question"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- Build the search index (runs once when the file is imported) ------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")   # turns text into 384 numbers
chunks = load_all_chunks()

client = chromadb.Client()                           # in-memory vector database
collection = client.create_collection("policies", metadata={"hnsw:space": "cosine"})
collection.add(
    ids=[str(i) for i in range(len(chunks))],
    documents=[c["text"] for c in chunks],
    embeddings=embedder.encode([c["text"] for c in chunks]).tolist(),
    metadatas=[{"source": c["source"], "section": c["section"]} for c in chunks],
)


def search(question, k=3):
    """Return the k chunks most similar to the question."""
    question_vector = embedder.encode([question]).tolist()
    result = collection.query(query_embeddings=question_vector, n_results=k)

    hits = []
    for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append({
            "text": text,
            "source": meta["source"],
            "section": meta["section"],
            "score": round(1 - distance, 3),   # cosine distance -> similarity
        })
    return hits


def ask_groq(question, hits):
    """Ask the Groq LLM to answer using ONLY the retrieved chunks. Returns None if no API key."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    from groq import Groq

    context = "\n\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, start=1))
    prompt = (
        "Answer the question using ONLY the context below. Be short (1-3 sentences) "
        "and cite passages like [1]. If the answer is not in the context, say you could not find it.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    try:
        response = Groq(api_key=api_key).chat.completions.create(
            model=GROQ_MODEL, temperature=0, max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as error:   # e.g. wrong key, no internet, model retired
        print("Groq call failed, using the best chunk instead:", error)
        return None


def answer(question, k=3):
    """Full pipeline: route the question, search, and answer."""
    # 1) greetings
    if question.lower().strip().startswith(("hi", "hello", "hey", "thanks")):
        return {"intent": "smalltalk", "answer": "Hello! Ask me about leave, expenses, remote work or data security.", "sources": []}

    # 2) search
    hits = search(question, k)

    # 3) if even the best chunk is not similar enough, it is not a policy question
    if hits[0]["score"] < MIN_SCORE:
        return {"intent": "out_of_scope", "answer": "I can only answer questions about the company policy documents.", "sources": []}

    # 4) answer with the LLM, or fall back to the best chunk
    llm_answer = ask_groq(question, hits)
    return {
        "intent": "policy",
        "answer": llm_answer or hits[0]["text"],
        "sources": [{"source": h["source"], "section": h["section"], "score": h["score"]} for h in hits],
    }
