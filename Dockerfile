# Image de base légère avec Python
FROM python:3.9-slim

# Répertoire de travail dans le conteneur
WORKDIR /app

# Copie des fichiers nécessaires
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste du projet
COPY . .

# Exposition du port utilisé par FastAPI
EXPOSE 8000

# Commande pour lancer l'API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]