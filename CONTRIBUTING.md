# Contribuer à SynaptiQ

Les contributions doivent préserver trois propriétés: isolation des données, idempotence des événements et explicabilité du retrieval.

1. Créer une branche ciblée et ajouter les tests concernés.
2. Exécuter `ruff check apps packages scripts tests` et `pytest tests/unit -q`.
3. Pour une évolution de schéma, créer une migration Alembic non destructive et documenter la procédure de mise à niveau.
4. Toute évolution Q-EM doit inclure un résultat de benchmark contre le retrieval vectoriel de référence.

Les issues de sécurité ne doivent pas être publiées publiquement: suivre `SECURITY.md`.
