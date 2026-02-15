# 📊 Projet 8 - Système de Scoring de Crédit

> **Prédiction automatisée de l'accord/refus de crédits avec API REST et monitoring en temps réel**

---

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture du projet](#architecture-du-projet)
- [Technologies utilisées](#technologies-utilisées)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Structure des données](#structure-des-données)
- [API REST](#api-rest)
- [Monitoring et logs](#monitoring-et-logs)
- [Déploiement](#déploiement)
- [Tests](#tests)
- [Documentation supplémentaire](#documentation-supplémentaire)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Vue d'ensemble

Ce projet implémente un **système complet de scoring de crédit** permettant de prédire l'accord ou le refus d'un prêt bancaire pour un client. Le système combine :

✅ **Modélisation ML avancée** avec LightGBM
✅ **API REST moderne** avec FastAPI
✅ **Interface de monitoring** avec Streamlit
✅ **Versioning du modèle** via MLflow et Hugging Face Hub
✅ **Containerisation** avec Docker
✅ **Tests automatisés** avec pytest

**Objectif principal** : Prédire si un client pourra rembourser son prêt en fonction de ses caractéristiques financières et personnelles.

---

## 🏗️ Architecture du projet

### Vue d'ensemble architecture

```
┌─────────────────────────────────────────────────────────┐
│                      UTILISATEURS                        │
└──────────────┬──────────────────────────┬─────────────────┘
               │                          │
         ┌─────▼─────┐            ┌──────▼──────┐
         │ Streamlit  │            │   FastAPI   │
         │ Dashboard  │            │  API REST   │
         └─────┬──────┘            └──────┬──────┘
               │                          │
         ┌─────▼──────────────────────────▼─────┐
         │     Modèle LightGBM (en production)   │
         │         (version contrôlée)          │
         └─────┬──────────────────────────┬─────┘
               │                          │
         ┌─────▼──────┐            ┌──────▼──────┐
         │ MLflow DB  │            │  Hugging    │
         │ (versioning)│            │  Face Hub   │
         └─────────────┘            └─────────────┘
```

### Répertoires clés

```
projet8/
├── src/                        # Code source principal
│   ├── api/
│   │   └── main.py            # API FastAPI (endpoints)
│   ├── model/
│   │   └── model.py           # Chargement du modèle
│   └── inference/             # Module inférence
├── notebooks/                 # Notebooks Jupyter
│   ├── 01_eda.ipynb          # Analyse exploratoire des données
│   ├── 02_fusion.ipynb       # Fusion et préparation des données
│   └── 03_modelisation.ipynb # Entraînement du modèle
├── Data/                      # Données brutes et traitées
│   ├── features_clients.csv   # Caractéristiques clients
│   └── Processed/
│       └── application_train_fused.csv  # Données fusionnées
├── app/
│   └── model.joblib           # Modèle sérialisé
├── tests/                     # Tests automatisés
├── monitoring/               # Monitoring et analyse
│   └── (contient les scripts de monitoring)
├── mlruns/                   # Artefacts MLflow
│   └── (versioning du modèle)
├── docker/                   # Configuration Docker
├── scripts/                  # Scripts utilitaires
│   └── export_model.py      # Export du modèle
├── streamlit_app.py          # Dashboard de monitoring Streamlit
├── drift_analysis.py         # Analyse de data drift
├── analyze_logs.py          # Analyse des logs API
├── debug_model.py            # Script de débogage
├── Dockerfile                # Configuration Docker
├── mlflow.db                 # Base de données MLflow
├── api_logs.jsonl           # Logs des prédictions API
├── data_drift_report.html   # Rapport de drift Evidently
└── requirements.txt          # Dépendances Python
```

---

## 🛠️ Technologies utilisées

| Catégorie | Technologies |
|-----------|-------------|
| **ML/Data Science** | LightGBM, scikit-learn, pandas, numpy |
| **Web Backend** | FastAPI, Uvicorn |
| **Monitoring** | Streamlit, MLflow, Evidently.ai |
| **Data Drift Detection** | Evidently.ai (rapports HTML) |
| **Versioning Modèle** | MLflow, Hugging Face Hub |
| **Testing** | pytest, httpx |
| **Containerisation** | Docker |
| **Python Version** | 3.12 (compatible 3.9+) |

---

## 📦 Installation

### 1. Cloner le repositorysion

```bash
git clone <repository-url>
cd projet8
```

### 2. Créer un environnement virtuel

```bash
# Windows
python -m venv projet8
projet8\Scripts\activate

# MacOS/Linux
python -m venv projet8
source projet8/bin/activate
```

### 3. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Vérifier l'installation

```bash
python debug_model.py
```

---

## ⚙️ Configuration

### Variables d'environnement requises

Créez un fichier `.env` à la racine du projet :

```bash
# Hugging Face Hub (optionnel, pour le téléchargement du modèle)
HF_TOKEN=hf_votre_token_huggingface

# MLflow (optionnel)
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

### Obtenir votre token HF

1. Créez un compte sur [huggingface.co](https://huggingface.co)
2. Allez dans Settings → Access Tokens
3. Créez un nouveau token
4. Collez-le dans votre `.env`

### Chargement du modèle

Le modèle se charge automatiquement de 3 sources (dans cet ordre) :

1. **Hugging Face Hub** : `PCelia/credit-scoring-model`
2. **MLflow local** : `models:/CreditScoring_LightGBM/Production`
3. **Fichier local** : `app/model.joblib`

---

## 🚀 Utilisation

### Lancer l'API

```bash
# Mode développement
uvicorn src.api.main:app --reload

# Mode production
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

L'API sera accessible sur `http://localhost:8000`

### Accéder à la documentation interactive

- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

### Lancer le tableau de bord Streamlit

```bash
streamlit run streamlit_app.py
```

Accessible sur `http://localhost:8501`

---

## 📊 Structure des données

### Fichiers de données

| Fichier | Description | Taille |
|---------|------------|--------|
| `features_clients.csv` | Caractéristiques brutes des clients | Variable |
| `application_train_fused.csv` | Données d'entraînement fusionnées et nettoyées | Variable |

### Colonnes principales

Les données contiennent des informations sur :
- **Identité** : SK_ID_CURR (identifiant client)
- **Données personnelles** : age, genre, situation familiale
- **Données financières** : revenus, dettes existantes, historique de crédit
- **Données professionnelles** : secteur, durée d'emploi

### Pipeline de données

```
Données brutes
    ↓
01_eda.ipynb (Exploration)
    ↓
02_fusion.ipynb (Fusion et nettoyage)
    ↓
Données traitées
    ↓
03_modelisation.ipynb (Entraînement LightGBM)
    ↓
Modèle ML exporté
```

---

## 🔌 API REST

### Endpoints disponibles

#### 1. **Prédiction (POST)**

```bash
POST /predict
Content-Type: application/json

{
  "sk_id_curr": 100001
}
```

**Réponse succès (200)**
```json
{
  "sk_id_curr": 100001,
  "score": 0.73,
  "decision": "ACCORD",
  "probability_refusal": 0.27,
  "probability_approval": 0.73
}
```

**Réponse erreur (404)**
```json
{
  "detail": "Client not found"
}
```

#### 2. **Santé de l'API (GET)**

```bash
GET /health
```

**Réponse (200)**
```json
{
  "status": "healthy"
}
```

### Exemplez d'utilisation

#### Avec `curl`

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"sk_id_curr\": 100001}"
```

#### Avec Python

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"sk_id_curr": 100001}
)
result = response.json()
print(f"Score: {result['score']}")
print(f"Décision: {result['decision']}")
```

#### Avec `httpx` (async)

```python
import httpx
import asyncio

async def get_prediction():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/predict",
            json={"sk_id_curr": 100001}
        )
        return response.json()

asyncio.run(get_prediction())
```

### Codes HTTP

| Code | Signification |
|------|---------------|
| 200 | Prédiction réussie |
| 404 | Client non trouvé |
| 422 | Format de requête invalide |
| 500 | Erreur serveur |

---

## 📈 Monitoring et logs

### Tableau de bord Streamlit

Lancez le tableau de bord de monitoring :

```bash
streamlit run streamlit_app.py
```

L'application `streamlit_app.py` fournit en temps réel :

- 📊 **Latence API** : Métrique et graphique des temps de réponse
- 📉 **Distribution des scores** : Analyse des décisions de crédit
- 💾 **Historique complet** : Tous les appels enregistrés en temps réel
- 🔍 **Data Drift** : Surveillance de la dérive des données avec Evidently
- 🎯 **Statut du système** : CPU et mémoire en temps réel

Accessible sur http://localhost:8501

### Analyse du Data Drift

**Evidently.ai** est intégré pour détecter la dérive des données en temps réel :

#### Génération de rapports

```bash
# Générer un rapport de drift
python monitoring/drift_analysis.py
```

Cela génère `data_drift_report.html` avec :
- ✅ Détection automatique des dérives
- ✅ Comparaison des distributions (référence vs. données actuelles)
- ✅ Alertes sur les changements significatifs
- ✅ Graphiques détaillés par feature

#### Analyse interactive

Vous pouvez aussi utiliser le notebook interactif :

```bash
jupyter notebook data_drift_analysis.ipynb
```

Ce notebook permet de :
- Explorer les dérives en temps réel
- Configurer les seuils d'alerte personnalisés
- Générer des rapports HTML automatiques
- Visualiser les changements de distribution

### Format des logs

Les logs sont stockés dans `api_logs.jsonl` (JSON Lines) :

```json
{"timestamp": "2024-02-08T10:30:45", "sk_id_curr": 100001, "score": 0.73, "total_time": 0.045}
{"timestamp": "2024-02-08T10:30:50", "sk_id_curr": 100002, "score": 0.42, "total_time": 0.038}
```

### Visualiser les logs

```bash
# Voir les 10 dernières prédictions
tail -10 api_logs.jsonl

# Convertir en CSV pour analyse
python analyze_logs.py
```

### MLflow Tracking

Les expériences de modélisation sont tracées avec MLflow :

```bash
# Consulter l'historique des modèles
mlflow ui

# Accéder à http://localhost:5000
```

---

## 🐳 Déploiement

### Avec Docker (recommandé)

#### 1. Construire l'image

```bash
docker build -t credit-scoring:latest .
```

#### 2. Exécuter le conteneur

```bash
# Mode développement avec volumes
docker run -p 8000:7860 \
  -e HF_TOKEN=hf_votre_token \
  -v $(pwd)/Data:/app/Data \
  -v $(pwd)/api_logs.jsonl:/app/api_logs.jsonl \
  credit-scoring:latest

# Mode production
docker run -d -p 8000:7860 \
  --name credit-api \
  -e HF_TOKEN=hf_votre_token \
  credit-scoring:latest
```

#### 3. Accéder à l'API

```
http://localhost:8000
```

#### 4. Monitorer le conteneur

```bash
# Voir les logs
docker logs credit-api

# Accéder à Streamlit (dans le conteneur)
docker exec credit-api streamlit run streamlit_app.py --server.port 8501
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:7860"
    environment:
      - HF_TOKEN=${HF_TOKEN}
    volumes:
      - ./Data:/app/Data
      - ./api_logs.jsonl:/app/api_logs.jsonl
```

Lancer avec :
```bash
docker-compose up
```

### Déploiement sur Hugging Face Spaces

Ce projet est configuré pour Hugging Face Spaces (voir `README_HF.md`) :

```yaml
title: Pret A Depenser
emoji: 📉
colorFrom: green
colorTo: indigo
sdk: docker
```

---

## 🧪 Tests

### Structure des tests

```
tests/
├── unit/                    # Tests unitaires
│   ├── test_model_unit.py          # Tests du modèle
│   ├── test_preprocessing.py       # Tests du prétraitement
│   ├── test_input_validation.py    # Validation des entrées
│   └── test_model_loading.py       # Chargement du modèle
├── fonctionnal/             # Tests fonctionnels/intégration
│   ├── test_api.py                 # Tests de l'API REST
│   ├── test_response_schema.py     # Schéma des réponses
│   ├── test_error_handling.py      # Gestion des erreurs
│   └── test_latency.py             # Latence des réponses
└── conftest.py              # Configurations pytest
```

### Lancer les tests

```bash
# Tous les tests
pytest

# Avec verbose
pytest -v

# Coverage (couverture de code)
pytest --cov=src

# Seulement tests unitaires
pytest tests/unit/

# Seulement tests fonctionnels
pytest tests/fonctionnal/

# Test spécifique
pytest tests/unit/test_model_unit.py::test_model_prediction -v
```

### Exemples de tests

```bash
# Tests unitaires du modèle
pytest tests/unit/test_model_unit.py -v

# Tests API
pytest tests/fonctionnal/test_api.py -v

# Tests de latence
pytest tests/fonctionnal/test_latency.py -v

# Rapport coverage détaillé
pytest --cov=src --cov-report=html
```

### Ajouter vos propres tests

```python
# tests/unit/test_mon_test.py

import pytest
from src.model.model import load_model

def test_model_loading():
    """Teste le chargement du modèle"""
    model = load_model()
    assert model is not None

def test_prediction_shape():
    """Teste que la prédiction a la bonne forme"""
    model = load_model()
    predictions = model.predict([[1, 2, 3, 4, 5]])
    assert predictions.shape[0] == 1
```

---

## 📚 Documentation supplémentaire

### Notebooks Jupyter

| Notebook | Description |
|----------|-------------|
| [01_eda.ipynb](notebooks/01_eda.ipynb) | Analyse exploratoire des données (EDA) |
| [02_fusion.ipynb](notebooks/02_fusion.ipynb) | Fusion de sources et préparation |
| [03_modelisation.ipynb](notebooks/03_modelisation.ipynb) | Entraînement et validation du modèle || [data_drift_analysis.ipynb](data_drift_analysis.ipynb) | **NOUVEAU** : Analyse interactive du data drift avec Evidently |
### Scripts utilitaires

```bash
# Exporter le modèle depuis MLflow
python scripts/export_model.py

# Déboguer et tester le modèle
python debug_model.py

# Analyser les logs API en détail
python monitoring/analyze_logs.py

# Analyser la dérive des données (Data Drift)
python monitoring/drift_analysis.py
```

### Chaining des outils

Pipeline complet de monitoring :

```bash
# 1. Lancer l'API
uvicorn src.api.main:app --reload &

# 2. Générer quelques prédictions
for i in {1..10}; do
  curl -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d "{\"sk_id_curr\": $((100000 + i))}"
done

# 3. Analyser les logs
python monitoring/analyze_logs.py

# 4. Générer le rapport de drift
python monitoring/drift_analysis.py

# 5. Consulter le tableau de bord
streamlit run streamlit_app.py
```

### Ressources externes

- 📖 [Documentation FastAPI](https://fastapi.tiangolo.com)
- 📖 [Documentation LightGBM](https://lightgbm.readthedocs.io)
- 📖 [Documentation Streamlit](https://docs.streamlit.io)
- 📖 [Documentation MLflow](https://mlflow.org/docs)
- 📖 [Hub Hugging Face](https://huggingface.co)

---

## 🔧 Troubleshooting

### Problème : Modèle non trouvé

**Erreur**
```
FileNotFoundError: Impossible de charger le modèle (ni HF Hub, ni MLflow)
```

**Solutions**
```bash
# 1. Vérifier le chemin local
ls app/model.joblib

# 2. Vérifier MLflow
python -c "import mlflow; print(mlflow.get_tracking_uri())"

# 3. Définir le token HF
export HF_TOKEN=your_token
# ou dans .env
echo "HF_TOKEN=your_token" > .env
```

### Problème : Token HF expiré

**Solution**
```bash
# Créer un nouveau token sur https://huggingface.co/settings/tokens
# Mettre à jour le fichier .env
nano .env  # ou edit .env
```

### Problème : Port déjà utilisé

**Erreur**
```
Address already in use: ('0.0.0.0', 8000)
```

**Solution**
```bash
# Changer le port
uvicorn src.api.main:app --port 8001

# Ou tuer le processus existant
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Problème : Erreur LightGBM Windows

**Erreur**
```
ImportError: cannot open shared object file: No such file or directory
```

**Solution**
```bash
# Réinstaller LightGBM
pip uninstall lightgbm -y
pip install lightgbm --force-reinstall
```

### Problème : Streamlit ne se lance pas

**Solution**
```bash
# Vérifier les permissions
streamlit run streamlit_app.py --logger.level=debug

# Réinstaller Streamlit
pip uninstall streamlit -y
pip install streamlit
```

---

## 👥 Auteurs et contribution

Ce projet a été développé dans le cadre d'une formation en Machine Learning.

### Structure Git

```
main                    # Branche principale (production)
├── develop            # Branche de développement
└── feature/*          # Branches de fonctionnalités
```

### Contribuer

1. Créer une branche `feature/ma-feature`
2. Faire vos commits
3. Pousser vers le repo
4. Ouvrir une Pull Request

---

## 📄 Licence

Ce projet est à usage éducatif.

---

## 📞 Support

Pour toute question ou problème :
1. Consulter la section [Troubleshooting](#troubleshooting)
2. Vérifier les logs : `api_logs.jsonl`
3. Lancer les tests : `pytest -v`
4. Ouvrir une issue avec les détails

---

## ✨ Roadmap future

- [ ] ✅ **Data Drift Detection** (Evidently.ai) - COMPLÉTÉ
- [ ] ✅ **Monitoring Dashboard** (Streamlit) - COMPLÉTÉ
- [ ] ✅ **API Logging & Analytics** - COMPLÉTÉ
- [ ] Ajouter explication des prédictions (SHAP/LIME)
- [ ] Interface web avancée (React/Next.js)
- [ ] Alertes email sur data drift
- [ ] Améliorer le monitoring (Prometheus + Grafana)
- [ ] Déploiement Kubernetes
- [ ] Tests de performance E2E
- [ ] CI/CD pipeline GitHub Actions

---

---

## 📝 Nouveautés récentes

### v1.1.0 (Février 2026)

✨ **Nouvelles fonctionnalités** :
- 🔍 Détection automatique du **Data Drift** avec Evidently.ai
- 📊 Tableau de bord **Streamlit** pour le monitoring en temps réel
- 📈 Analyse des logs API avec **psutil** (CPU, mémoire)
- 📓 Notebook interactif pour l'analyse du drift
- 🚀 Support Docker amélioré avec volumes persistants

🐛 **Corrections** :
- Amélioration du chargement du modèle (fallback multi-sources)
- Meilleure gestion des erreurs API
- Optimisation des performances

---

**Dernière mise à jour** : 15 Février 2026