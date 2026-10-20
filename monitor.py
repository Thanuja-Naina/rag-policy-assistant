"""MONITORING: summarise the query log.  Run: python monitor.py

Watch for: many out_of_scope questions (users want topics your documents do not cover),
low top_score values (retrieval is struggling), or rising latency.
"""
import os

import pandas as pd

if not os.path.exists("logs/queries.csv"):
    raise SystemExit("No queries logged yet. Ask some questions through api.py or app.py first.")

log = pd.read_csv("logs/queries.csv")
print(f"Questions logged : {len(log)}")
print("Routing          :", log["intent"].value_counts().to_dict())
print(f"Rejected as out of scope: {(log['intent'] == 'out_of_scope').mean():.1%}")
print(f"Average latency  : {log['latency_ms'].mean():.0f} ms (slowest: {log['latency_ms'].max()} ms)")
policy = log[log["intent"] == "policy"]
if len(policy) > 0:
    print(f"Average retrieval score for policy questions: {policy['top_score'].astype(float).mean():.3f}")
    low = policy[policy["top_score"].astype(float) < 0.45]
    print(f"Policy questions with a weak match (score < 0.45): {len(low)}  <- review these")
