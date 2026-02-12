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
├── tests/                     # Tests unitaires
│   └── test_api.py           # Tests de l'API
├── docker/                    # Configuration Docker
├── scripts/                   # Scripts utilitaires
│   └── export_model.py       # Export du modèle
├── streamlit_app.py           # Application de monitoring
├── debug_model.py             # Script de débogage
├── Dockerfile                 # Configuration Docker
└── requirements.txt           # Dépendances Python
```

---

## 🛠️ Technologies utilisées

| Catégorie | Technologies |
|-----------|-------------|
| **ML/Data Science** | LightGBM, scikit-learn, pandas, numpy |
| **Web Backend** | FastAPI, Uvicorn |
| **Monitoring** | Streamlit, MLflow |
| **Versioning Modèle** | MLflow, Hugging Face Hub |
| **Testing** | pytest, httpx |
| **Containerisation** | Docker |
| **Python Version** | 3.9+ (3.12 dans l'environnement) |

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

L'application `streamlit_app.py` fournit :

- 📊 **Latence API** : Graphique en temps réel des temps de réponse
- 📉 **Distribution des scores** : Analyse des décisions de crédit
- 💾 **Historique complet** : Tous les appels enregistrés

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

---

## 🐳 Déploiement

### Avec Docker (recommandé)

#### 1. Construire l'image

```bash
docker build -t credit-scoring:latest .
```

#### 2. Exécuter le conteneur

```bash
docker run -p 8000:7860 \
  -e HF_TOKEN=hf_votre_token \
  credit-scoring:latest
```

#### 3. Accéder à l'API

```
http://localhost:8000
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

### Lancer les tests

```bash
# Tous les tests
pytest

# Avec verbose
pytest -v

# Coverage
pytest --cov=src

# Test spécifique
pytest tests/test_api.py::test_prediction -v
```

### Tests disponibles

```python
# tests/test_api.py

✓ test_predict_valid_client()        # Prédiction client valide
✓ test_predict_invalid_client()      # Client inexistant
✓ test_predict_invalid_format()      # Format JSON invalide
✓ test_health_check()                # Vérification santé API
```

### Ajouter vos propres tests

```python
# tests/test_api.py

def test_mon_test():
    """Description du test"""
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={"sk_id_curr": 100001}
    )
    assert response.status_code == 200
```

---

## 📚 Documentation supplémentaire

### Notebooks Jupyter

| Notebook | Description |
|----------|-------------|
| [01_eda.ipynb](notebooks/01_eda.ipynb) | Analyse exploratoire des données (EDA) |
| [02_fusion.ipynb](notebooks/02_fusion.ipynb) | Fusion de sources et préparation |
| [03_modelisation.ipynb](notebooks/03_modelisation.ipynb) | Entraînement et validation du modèle |

### Scripts utilitaires

```bash
# Exporter le modèle
python scripts/export_model.py

# Déboguer le modèle
python debug_model.py

# Analyser les logs
python analyze_logs.py
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

- [ ] Ajouter explication des prédictions (SHAP)
- [ ] Interface web avancée (React)
- [ ] Améliorer le monitoring (Prometheus)
- [ ] Déploiement Kubernetes
- [ ] Tests de performance
- [ ] CI/CD pipeline complet

---

**Dernière mise à jour** : Février 2026