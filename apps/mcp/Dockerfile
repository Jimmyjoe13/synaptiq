# SynaptiQ serveur MCP — build depuis la racine du repo (context: .)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY apps/mcp/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Utilisateur non privilégié (cf. Dockerfile de l'API).
RUN useradd --create-home --shell /usr/sbin/nologin synaptiq \
    && chown -R synaptiq:synaptiq /app
USER synaptiq

EXPOSE 8765
CMD ["python", "-u", "apps/mcp/server.py"]
