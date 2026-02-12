import json
import pandas as pd
import warnings
from evidently import Report
from evidently.presets import DataDriftPreset

warnings.filterwarnings("ignore")  # enlève le spam de warnings

# Charger logs
records = []
with open("api_logs.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

# Split référence / courant
reference_df = df.iloc[: len(df)//2][["score", "total_time"]]
current_df   = df.iloc[len(df)//2 :][["score", "total_time"]]

report = Report([DataDriftPreset()])

# IMPORTANT: le résultat (snapshot) porte save_html()
snapshot = report.run(reference_data=reference_df, current_data=current_df)
snapshot.save_html("evidently_report.html")

print("OK: evidently_report.html généré")