# SynaptiQ serveur MCP — build depuis la racine du repo (context: .)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Socle de dépendances PARTAGÉ avec l'API, le worker et le relais — corrigé le 11/08.
# `apps/mcp/requirements.txt` ne portait que fastmcp/requests/python-dotenv, or la chaîne
# d'import de `synaptiq_core` (server.py:19) charge `numpy` : l'image échouait au démarrage
# sur `ModuleNotFoundError: No module named 'numpy'`. C'était la récidive exacte du bug F19
# (prometheus_client sur le worker le 29/07), parce que le MCP était le SEUL service à ne
# pas consommer le socle commun. Le socle ne fait que quelques Mo de plus — même ordre de
# grandeur que fastmcp — et il supprime la classe de dérive entière.
COPY requirements-common.txt ./
RUN pip install --no-cache-dir -r requirements-common.txt

COPY apps/mcp/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Utilisateur non privilégié (cf. Dockerfile de l'API).
RUN useradd --create-home --shell /usr/sbin/nologin synaptiq \
    && chown -R synaptiq:synaptiq /app
USER synaptiq

EXPOSE 8765
CMD ["python", "-u", "apps/mcp/server.py"]
