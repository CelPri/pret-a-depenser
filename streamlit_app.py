import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import psutil
from datetime import datetime

st.title("Monitoring API – Credit Scoring")


# Chargement des logs


records = []
with open("api_logs.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

# Conversion timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])


# Latence API (temps total)


st.subheader("Latence API (temps total)")

st.metric(
    "Latence moyenne (ms)",
    round(df["total_time"].mean() * 1000, 2)
)

st.metric(
    "Latence max (ms)",
    round(df["total_time"].max() * 1000, 2)
)

st.line_chart(df["total_time"] * 1000)


# Temps d'inférence modèle


st.subheader("Temps d'inférence modèle")

st.metric(
    "Temps moyen (ms)",
    round(df["inference_time"].mean() * 1000, 2)
)

st.metric(
    "Temps max (ms)",
    round(df["inference_time"].max() * 1000, 2)
)

fig, ax = plt.subplots()

ax.plot(df["inference_time"].values * 1000)

# Point rouge dernière requête
ax.scatter(
    len(df) - 1,
    df["inference_time"].iloc[-1] * 1000,
    color="red",
    s=80
)

ax.set_xlabel("Requête")
ax.set_ylabel("Inference time (ms)")

st.pyplot(fig)


# Distribution des scores


st.subheader("Distribution des scores")

st.metric(
    "Score moyen",
    round(df["score"].mean(), 4)
)

st.bar_chart(df["score"])


# Requêtes par minute


requests_per_min = (
    df.set_index("timestamp")
      .resample("1min")
      .size()
)

st.subheader("Requêtes par minute")
st.line_chart(requests_per_min)


# Utilisation CPU et RAM

st.subheader("Utilisation système")

cpu_usage = psutil.cpu_percent(interval=None)
ram_usage = psutil.virtual_memory().percent

col1, col2 = st.columns(2)

#col1.metric("CPU usage (%)", cpu_usage)
col2.metric("RAM usage (%)", ram_usage)

# Dernière requête

last_request = df.iloc[-1]
st.write("Dernière requête :", last_request["timestamp"])