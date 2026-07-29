# SynaptiQ API — build depuis la racine du repo (context: .)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/packages/core:/app/packages/sdk-python

WORKDIR /app

# curl requis pour le healthcheck docker-compose
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Installer les deps d'abord (cache Docker)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code (l'API importe apps.worker et packages/core)
COPY . .

# Utilisateur non privilégié : une exécution en root donnerait à une éventuelle exécution
# de code arbitraire les pleins pouvoirs dans le conteneur (et sur les volumes montés).
# `alembic` doit rester exécutable par cet utilisateur : le service Compose `migrate`
# réutilise cette même image.
RUN useradd --create-home --shell /usr/sbin/nologin synaptiq \
    && chown -R synaptiq:synaptiq /app
USER synaptiq

EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
