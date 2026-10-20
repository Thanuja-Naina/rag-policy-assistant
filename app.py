"""A small chat demo.  Run:  streamlit run app.py     (this file is what you deploy on Hugging Face Spaces)"""
import time

import streamlit as st

from logger import log_query
from rag import answer

st.title("Company Policy Assistant")
st.write("Ask about leave, expenses, remote work or data security. (Sample fictional policies.)")

question = st.text_input("Your question", "How many days of annual leave do I get?")
if st.button("Ask") and question:
    start = time.time()
    result = answer(question)
    log_query(question, result, (time.time() - start) * 1000)
    st.write(result["answer"])
    if result["sources"]:
        st.caption("Sources: " + ", ".join(f"{s['source']} - {s['section']} (score {s['score']})" for s in result["sources"]))
    else:
        st.caption(f"Routed as: {result['intent']}")
