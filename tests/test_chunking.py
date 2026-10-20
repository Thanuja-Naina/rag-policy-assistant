import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chunking import load_all_chunks, split_into_chunks


def test_split_into_chunks():
    text = "# Title\n\n## A\nOne. Two.\n\n## B\nThree."
    chunks = split_into_chunks(text, "x.md")
    assert [c["section"] for c in chunks] == ["A", "B"]
    assert chunks[0]["text"].startswith("Title - A:")


def test_every_eval_question_points_to_a_real_section():
    real = {(c["source"], c["section"]) for c in load_all_chunks()}
    for q in json.load(open("eval/eval_set.json"))["in_scope"]:
        assert (q["expected_source"], q["expected_section"]) in real


def test_query_logger_writes_rows(tmp_path):
    import csv
    from logger import log_query
    log_file = str(tmp_path / "logs" / "q.csv")
    log_query("How many leave days?", {"intent": "policy", "sources": [{"score": 0.71}]}, 120.4, log_file)
    log_query("hello", {"intent": "smalltalk", "sources": []}, 3.2, log_file)
    rows = list(csv.reader(open(log_file, encoding="utf-8")))
    assert rows[0] == ["time", "question", "intent", "top_score", "latency_ms"]
    assert len(rows) == 3 and rows[1][2] == "policy" and rows[2][3] == ""
