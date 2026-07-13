# SynaptiQ 🧠💫

> **Second Cerveau Sémantique et Temporel pour Agents IA Autonomes**

SynaptiQ est une infrastructure de mémoire à long terme (LTM) avancée conçue spécifiquement pour les agents IA autonomes (comme Ouroboros George). Il s'agit d'une alternative ultra-performante aux outils de gestion de connaissances traditionnels (Obsidian, Mempalace) qui repose sur l'indexation vectorielle sémantique et des processus d'intrication conceptuelle (**Quantum-like Entanglement Memory - Q-EM**).

---

## 🌌 Vision & Philosophie

Contrairement aux outils de prise de notes humains qui imposent des arborescences ou des liens markdown rigides créés à la main, **SynaptiQ** permet aux agents IA d'auto-indexer leur flux d'expérience de manière multidimensionnelle. 

Grâce à l'intrication vectorielle sémantique, l'agent peut faire des analogies non-linéaires en reliant des expériences distantes dans le temps mais connectées par le sens, ce qui lui évite de répéter ses erreurs et augmente sa pertinence décisionnelle en production.

---

## 🏗️ Architecture du Système

SynaptiQ est conçu avec une architecture modulaire asynchrone pour garantir une exécution sans aucune latence lors des appels de l'agent principal :

```mermaid
graph TD
    Agent[Agent Ouroboros George] -->|1. Chat & Logs Event| API[SynaptiQ API : FastAPI]
    API -->|2. Push Job| Redis[Message Broker : Redis]
    Redis -->|3. Pull Job| Worker[Consolidation Worker]
    Worker -->|4. Indexation & Vectorisation| DB[(Database: PostgreSQL + pgvector)]
    DB -->|5. Recall & Réhydratation| API
    API -->|6. Context Block system prompt| Agent
```

1. **SynaptiQ API (FastAPI - Port 8000) :** Reçoit les événements de chat (`/events`) et fournit dynamiquement le contexte de prompt réhydraté (`/context/build`).
2. **Message Broker (Redis) :** File d'attente asynchrone de traitement des souvenirs pour isoler l'agent de tout traitement lourd.
3. **Consolidation Worker (Python) :** Récupère les jobs dans Redis, calcule les embeddings de souvenirs, et crée les associations d'intrication dans la base de données.
4. **Base de Données (PostgreSQL & pgvector) :** Stockage hybride relationnel (métadonnées, logs d'outils, chronologie) et vectoriel (embeddings sémantiques).

---

## ⚡ Fonctionnalités Clés

* **Event-Driven Memory Capture :** Abonnement automatique aux bus d'événements de l'agent (`chat.outbound`) pour capturer chaque message de façon transparente.
* **Dynamic Context Rehydration :** Injection automatique de souvenirs pertinents (sémantiques, procéduraux, épisodiques) directement dans le prompt système du LLM de l'agent avant chaque tâche.
* **Intrication Conceptuelle (Q-EM) :** Cartographie et liaison de souvenirs reliés par le sens, facilitant des sauts conceptuels de haut niveau.
* **Conception Robuste Windows & Linux :** Adapté aux verrous asynchrones complexes grâce à un système de polling résistant aux timeouts sockets.

---

## 🚀 Installation et Démarrage Local

### Prérequis
* Docker & Docker Compose
* Python 3.11+
* `uv` (recommandé pour la gestion rapide de l'environnement virtuel)

### 1. Cloner et Initialiser l'Environnement
```bash
git clone https://github.com/Jimmyjoe13/synaptiq.git
cd synaptiq
```

Configurez votre fichier `.env` à la racine (voir `.env.example` si présent) :
```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/synaptiq
REDIS_URL=redis://localhost:6379/0
SYNAPTIQ_API_URL=http://127.0.0.1:8000
```

### 2. Démarrer l'Infrastructure de Données
Lancez PostgreSQL (avec l'extension pgvector) et Redis à l'aide de Docker Compose :
```bash
docker-compose up -d
```

### 3. Installer les Dépendances Python
```bash
# Avec uv (recommandé)
uv pip install -r requirements.txt

# Ou avec pip standard
pip install -r requirements.txt
```

### 4. Lancer le Serveur API SynaptiQ
Le serveur API écoute par défaut sur le port `8000` :
```bash
python -m uvicorn apps.api.main:app --reload --port 8000
```

### 5. Lancer le Worker de Consolidation
Le worker écoute Redis et traite les souvenirs en arrière-plan :
```bash
python apps/worker/worker.py
```

---

## 🔌 Exemple d'Intégration dans un Agent (Ouroboros)

### Étape 1 : Capturer les Messages Sortants
Le connecteur s'abonne au bus d'événements de l'agent et envoie un événement d'indexation à SynaptiQ :
```python
import requests

def capture_chat_interaction(task_id: str, message: str):
    payload = {
        "event_type": "chat_outbound",
        "task_id": task_id,
        "content": message
    }
    requests.post("http://127.0.0.1:8000/events", json=payload, timeout=2.0)
```

### Étape 2 : Réhydrater la Mémoire au Démarrage de la Tâche
Avant d'envoyer la requête au LLM, l'agent interroge l'API SynaptiQ pour obtenir les souvenirs contextuels et les formate en Markdown dans le prompt système :
```python
def format_memory_for_prompt(memory_context: dict) -> str:
    if not memory_context:
        return ""
    
    prompt = "\n## SynaptiQ Long-Term Memory (Q-EM)\n"
    prompt += "Voici des souvenirs pertinents de vos précédentes sessions :\n"
    for fact in memory_context.get("facts", []):
        prompt += f"- {fact}\n"
    return prompt
```

---

## 🧪 Tests Unitaires
Vous pouvez valider la suite de tests de communication à l'aide de `pytest` :
```bash
pytest tests/
```

---

## 📄 Licence
Ce projet est sous licence MIT. Pour plus d'informations, veuillez consulter le fichier `LICENSE`.
