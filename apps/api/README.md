# SynaptiQ API

Ce répertoire contient le serveur API principal de SynaptiQ basé sur FastAPI.

## Rôle
- Exposer les endpoints de collecte d'événements (`POST /events`).
- Exposer l'endpoint d'assemblage de contexte pour les agents (`POST /context/build`).
- Communiquer avec PostgreSQL (pgvector) pour la persistance et Redis pour le queueing.
