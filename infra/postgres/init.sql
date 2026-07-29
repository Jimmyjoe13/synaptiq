-- SynaptiQ — amorçage minimal de la base.
--
-- ⚠️ Ce fichier ne décrit PLUS le schéma. La seule autorité est Alembic (`migrations/`).
--
-- Jusqu'au 29/07, ce fichier et les migrations décrivaient le même schéma en parallèle :
-- toute évolution devait être écrite deux fois, la CI appliquait les deux à la suite, et
-- rejouer ce fichier sur une base existante échouait (l'ALTER TABLE ADD CONSTRAINT n'était
-- pas idempotent). Deux sources de vérité pour un schéma, c'est une divergence garantie.
--
-- Il ne reste ici que l'extension pgvector, créée par le point d'entrée Docker AVANT que
-- le service `migrate` ne tourne : la migration `0001_initial` la crée aussi
-- (IF NOT EXISTS), mais l'avoir dès l'initialisation du volume évite tout ordre de
-- démarrage douteux.
--
-- Créer ou faire évoluer le schéma :  alembic upgrade head
-- Nouvelle révision :                 alembic revision -m "..."  (SQL en IF NOT EXISTS)

CREATE EXTENSION IF NOT EXISTS vector;
