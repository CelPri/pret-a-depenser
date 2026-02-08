import json
import pandas as pd
import streamlit as st

st.title("Monitoring API – Credit Scoring")

# Charger les logs
records = []
with open("api_logs.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

st.subheader("Latence API")
st.write("Latence moyenne :", df["total_time"].mean())
st.write("Latence max :", df["total_time"].max())
st.line_chart(df["total_time"])

st.subheader("Distribution des scores")
st.write("Score moyen :", df["score"].mean())
st.bar_chart(df["score"])