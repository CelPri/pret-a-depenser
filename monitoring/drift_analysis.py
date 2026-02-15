import json
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

print("=== Reconstruction dataset production ===")

# --- 1. Charger données production depuis logs ---
inputs = []

with open("api_logs.jsonl") as f:
    for line in f:
        log = json.loads(line)
        if "inputs" in log:
            inputs.append(log["inputs"])

df_current = pd.DataFrame(inputs)

print("df_current shape:", df_current.shape)

if df_current.empty:
    raise ValueError("Aucune donnée production trouvée dans les logs.")

# --- 2. Charger référence ---
df_reference = pd.read_csv("Data/features_clients.csv")

if "SK_ID_CURR" in df_reference.columns:
    df_reference = df_reference.drop(columns=["SK_ID_CURR"])

print("df_reference shape:", df_reference.shape)

# --- 3. Aligner colonnes ---
common_cols = df_current.columns.intersection(df_reference.columns)

df_current = df_current[common_cols]
df_reference = df_reference[common_cols]

# Supprimer colonnes entièrement vides dans current
non_empty_cols = df_current.columns[df_current.notna().any()]

df_current = df_current[non_empty_cols]
df_reference = df_reference[non_empty_cols]

print("Colonnes finales utilisées :", len(non_empty_cols))

# --- 4. Échantillonner référence pour éviter biais taille ---
df_reference = df_reference.sample(n=len(df_current), random_state=42)

# --- 5. Simulation drift volontaire ---
df_current["AMT_INCOME_TOTAL"] *= 3
df_current["AMT_CREDIT"] *= 2
df_current["AMT_ANNUITY"] *= 2

# --- 6. Lancer Data Drift ---
print("=== Lancement Evidently ===")

report = Report(metrics=[DataDriftPreset()])

snapshot = report.run(
    reference_data=df_reference,
    current_data=df_current
)

snapshot.save_html("data_drift_report.html")

print("Rapport généré : data_drift_report.html")