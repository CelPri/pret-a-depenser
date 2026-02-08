import json
import pandas as pd

records = []
with open("api_logs.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

lat_mean = df["total_time"].mean()
lat_std = df["total_time"].std()
lat_max = df["total_time"].max()

df["is_anomaly"] = df["total_time"] > (lat_mean + 3 * lat_std)

print("latence moyenne:", lat_mean)
print("latence max:", lat_max)
print("anomalies:", df["is_anomaly"].sum())

# drift simple sur le score (proxy)
score_mean = df["score"].mean()
score_std = df["score"].std()

print("score moyen:", score_mean)
print("score std:", score_std)