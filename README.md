# GenAI Policy Support Assistant (RAG)

A question-answering API over company policy documents. It finds the most relevant policy sections using **sentence-transformer embeddings stored in ChromaDB**, rejects questions that are not about the policies, and uses **Groq (Llama)** to write a short answer with citations.

**Stack:** Python, FastAPI, ChromaDB, sentence-transformers (all-MiniLM-L6-v2), Groq API, Docker, pytest, GitHub Actions

> The policy files in `data/policies/` are **fictional samples** written for this project.

## Business problem
Employees repeatedly ask HR and IT the same policy questions, and searching long PDFs is slow. This assistant answers from the official documents only, shows its sources, and refuses questions it cannot support, so answers stay trustworthy.

## How it works

| File | What it does |
|---|---|
| `chunking.py` | Splits each policy file into one chunk per `## Heading` |
| `rag.py` | Embeds chunks with MiniLM, stores them in ChromaDB, searches, routes questions, asks Groq |

| `api.py` | FastAPI web API: `POST /ask` and `GET /health` (logs every question) |
| `app.py` | Streamlit chat demo (deploy this on Hugging Face Spaces) |
| `evaluate.py` | Measures search quality and speed on 28 test questions |
| `logger.py`, `monitor.py` | Monitoring: log every question, then summarise routing, latency and weak matches |

```
question -> embedding -> ChromaDB finds top-3 chunks -> Groq answers using ONLY those chunks
                |
                +-> similarity too low? -> "I can only answer policy questions"
```

## Run it
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY="your_key"        # optional; without it you get the best matching section as the answer
streamlit run app.py                  # web demo
uvicorn api:app --reload              # API: open http://127.0.0.1:8000/docs and try POST /ask
python monitor.py                     # after some questions: routing, latency, weak matches
```
Docker: `docker build -t policy-assistant . && docker run -p 8000:8000 -e GROQ_API_KEY=your_key policy-assistant`

**Live demo:** https://rag-policy-assistant-7hthyjtrytvjsezbvq7ywf.streamlit.app/

## Evaluation
`python evaluate.py` checks, for 28 paraphrased questions, whether the right policy file is found.
`hit@1` = it is the top result. `hit@3` = it is in the top 3.

| Method | hit@1 | hit@3 |
|---|---|---|
| Word-matching baseline (MD5 hashing) | 0.679 | 0.929 |
| MiniLM embeddings | _run `python evaluate.py` and paste here_ | |

Out-of-scope questions rejected correctly (MiniLM): _paste from the same run_
Average search time (MiniLM): _paste from the same run_

**Limitations**
- The baseline only matches identical words, which is why embeddings were added.
- `MIN_SCORE` in `rag.py` (0.30) is a starting value; adjust it using the evaluate output.
- The 28 questions were written by the author, so the results are indicative, not a benchmark.
- Evaluation checks search, not the quality of the LLM answers.

## Monitoring
Every question is saved to `logs/queries.csv` (time, question, routing, top score, latency). `python monitor.py` shows how many questions were rejected as out of scope (users may want topics your documents do not cover), how many matched weakly (documents may need updating), and the average response time.

## Tests
`pytest -q` checks the chunking code, the query logger, and that every test question points to a real policy section.
