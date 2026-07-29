# SynaptiQ Worker de consolidation — build depuis la racine du repo (context: .)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/packages/core

WORKDIR /app

# Socle de dépendances PARTAGÉ avec l'API et le relais.
# Le worker avait auparavant son propre `apps/worker/requirements.txt`, qui avait dérivé du
# code : il ne listait ni `prometheus-client` (importé depuis l'ajout du compteur
# d'extractions dégradées) ni `numpy`. L'image échouait au démarrage sur
# `ModuleNotFoundError: No module named 'prometheus_client'`.
# Le socle commun supprime cette classe de dérive sans embarquer FastAPI/uvicorn, dont le
# worker n'a aucun besoin.
COPY requirements-common.txt ./
RUN pip install --no-cache-dir -r requirements-common.txt

COPY . .

# Utilisateur non privilégié : une exécution en root donnerait à une éventuelle exécution
# de code arbitraire les pleins pouvoirs dans le conteneur (et sur les volumes montés).
RUN useradd --create-home --shell /usr/sbin/nologin synaptiq \
    && chown -R synaptiq:synaptiq /app
USER synaptiq

CMD ["python", "-u", "apps/worker/worker.py"]
