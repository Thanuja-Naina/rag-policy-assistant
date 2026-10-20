"""Step 4: measure how good the search is.

For each test question we know which policy file should be found.
  hit@1 = the right file is the top result
  hit@3 = the right file is in the top 3 results
We compare a simple word-matching baseline against the MiniLM embeddings.

Run:  python evaluate.py
"""
import hashlib
import json

import numpy as np

from chunking import load_all_chunks

questions = json.load(open("eval/eval_set.json"))["in_scope"]
outside_questions = json.load(open("eval/eval_set.json"))["out_of_scope"]
chunks = load_all_chunks()

STOP_WORDS = {"a", "an", "the", "is", "are", "was", "to", "of", "in", "on", "for", "and", "or", "do", "does",
              "i", "my", "me", "we", "you", "your", "it", "can", "how", "what", "when", "which", "with", "at",
              "by", "from", "as", "if", "get", "much", "be"}


def hash_embed(text, size=512):
    """BASELINE: turn each word into a position using an MD5 hash, then count words.
    It only matches identical words, so it cannot understand paraphrases."""
    vector = np.zeros(size)
    for word in text.lower().replace("?", "").replace(".", "").replace(",", "").split():
        if word in STOP_WORDS:
            continue
        number = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vector[number % size] += 1 if (number >> 64) & 1 else -1
    length = np.linalg.norm(vector)
    return vector / length if length > 0 else vector


def search_hash(question, k=3):
    chunk_vectors = np.array([hash_embed(c["text"]) for c in chunks])
    scores = chunk_vectors @ hash_embed(question)          # cosine similarity (vectors are normalised)
    best = np.argsort(-scores)[:k]
    return [chunks[i]["source"] for i in best]


def hit_rates(search_function):
    hit1 = hit3 = 0
    for q in questions:
        sources = search_function(q["question"])
        hit1 += sources[0] == q["expected_source"]
        hit3 += q["expected_source"] in sources
    return round(hit1 / len(questions), 3), round(hit3 / len(questions), 3)


print(f"Questions: {len(questions)}")
h1, h3 = hit_rates(search_hash)
print(f"Hash baseline : hit@1 = {h1}  hit@3 = {h3}")

try:
    from rag import MIN_SCORE, search     # needs sentence-transformers + chromadb installed

    h1, h3 = hit_rates(lambda q: [h["source"] for h in search(q, 3)])
    print(f"MiniLM        : hit@1 = {h1}  hit@3 = {h3}")
    correct = sum(search(q, 1)[0]["score"] < MIN_SCORE for q in outside_questions)
    print(f"Out-of-scope questions correctly rejected: {correct}/{len(outside_questions)}")
    import time
    start = time.time()
    for q in questions:
        search(q["question"], 3)
    print(f"Average search time: {(time.time() - start) / len(questions) * 1000:.0f} ms per question")
except ImportError:
    print("Install requirements.txt to evaluate the MiniLM embeddings too.")
