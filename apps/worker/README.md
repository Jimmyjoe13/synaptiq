# SynaptiQ Worker

Ce répertoire contient le worker asynchrone d'extraction et de classification de mémoire.

## Rôle
- Écouter la queue d'événements bruts sur Redis.
- Extraire de manière autonome les faits, préférences et règles via LLM.
- Générer les embeddings sémantiques.
- Consolider les nouvelles mémoires dans PostgreSQL.
