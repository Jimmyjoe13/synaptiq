# SynaptiQ — points d'entrée du dépôt.
#
# `make` n'est pas installé par défaut sur Windows : l'équivalent PowerShell de chaque
# cible est dans `scripts/dev.ps1` (mêmes noms). Ce Makefile reste la référence, c'est ce
# qu'un contributeur cherche en premier en arrivant sur un dépôt.

PY ?= python
DB_URL ?= postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db
REDIS_URL_LOCAL ?= redis://127.0.0.1:6399/0

.PHONY: help install up down migrate lint types test test-unit test-integration \
        coverage bench bench-explain clean

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Installe les dépendances de développement
	$(PY) -m pip install -r requirements-dev.txt

up:  ## Démarre l'infrastructure locale (Postgres + Redis)
	docker compose up -d postgres redis

down:  ## Arrête l'infrastructure locale
	docker compose down

migrate:  ## Applique les migrations (Alembic est la seule autorité du schéma)
	DATABASE_URL=$(DB_URL) alembic upgrade head

lint:  ## Ruff (la sélection de règles vit dans ruff.toml)
	ruff check apps packages scripts tests benchmarks migrations examples

types:  ## Mypy sur packages/core (cf. mypy.ini)
	PYTHONPATH=packages/core mypy

test-unit:  ## Tests unitaires (aucune infrastructure requise)
	$(PY) -m pytest tests/unit -q

test-integration:  ## Tests d'intégration (exige `make up` et `make migrate`)
	EMBEDDING_PROVIDER=mock DATABASE_URL=$(DB_URL) REDIS_URL=$(REDIS_URL_LOCAL) \
		$(PY) -m pytest -m integration -q

test: test-unit test-integration  ## Suite complète

coverage:  ## Couverture de packages/core, avec le seuil de la CI
	PYTHONPATH=packages/core $(PY) -m pytest tests/unit -q \
		--cov=synaptiq_core --cov-report=term-missing --cov-fail-under=90

# ─── Mesures ────────────────────────────────────────────────────────────────
#
# ⚠️ `bench` exige un endpoint LLM joignable (extraction + juge) et prend plusieurs heures
# sur les 10 conversations. Les résultats ne sont exploitables qu'à cette échelle :
# sur une seule conversation (~152 questions), l'intervalle de confiance à 95 % vaut
# ~±8 points et aucun écart de quelques points n'est significatif (cf. synaptiq_core.stats).

DATASET ?= benchmarks/locomo10.json
CONV ?= all
BENCH_OUT ?= benchmarks/results_$(CONV).json

bench:  ## Benchmark LOCOMO complet (10 conversations, IC à 95 % publiés)
	EMBEDDING_PROVIDER=lmstudio DATABASE_URL=$(DB_URL) REDIS_URL=$(REDIS_URL_LOCAL) \
		$(PY) benchmarks/locomo_runner.py $(DATASET) \
			--conv $(CONV) --arm both --qa-workers 4 --out $(BENCH_OUT)

bench-explain:  ## Vérifie que les requêtes de retrieval utilisent bien l'index HNSW
	DATABASE_URL=$(DB_URL) $(PY) scripts/explain_retrieval.py --seed 5000

clean:  ## Supprime les caches locaux
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
