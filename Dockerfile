# Dockerfile - SenSante
# Image de base : Python 3.11 leger
FROM python:3.11-slim

# Dossier de travail dans le conteneur
WORKDIR /app

# Copier et installer les dependances d'abord
# (optimisation du cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code du projet
COPY . .

# Declarer le port
EXPOSE 8000

# Commande de demarrage
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]