"""Web API.  Run:  uvicorn api:app --reload   then open http://127.0.0.1:8000/docs"""
import time

from fastapi import FastAPI
from pydantic import BaseModel

from logger import log_query
from rag import answer

app = FastAPI(title="Policy Support Assistant")


class Question(BaseModel):      # the JSON body we expect: {"question": "..."}
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(body: Question):
    start = time.time()
    result = answer(body.question)
    log_query(body.question, result, (time.time() - start) * 1000)     # monitoring
    return result
