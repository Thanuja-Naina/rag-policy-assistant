"""MONITORING helper: save every question, its routing and the response time to a CSV file."""
import csv
import os
from datetime import datetime

LOG_FILE = "logs/queries.csv"


def log_query(question, result, latency_ms, log_file=LOG_FILE):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    is_new_file = not os.path.exists(log_file)
    top_score = result["sources"][0]["score"] if result["sources"] else ""
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["time", "question", "intent", "top_score", "latency_ms"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), question, result["intent"], top_score, round(latency_ms)])
