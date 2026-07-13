import os
import sys
import time

# Ajouter le SDK local au path pour les besoins de la démo
sys.path.append(os.path.join(os.path.dirname(__file__), "../packages/sdk-python"))

from synaptiq_sdk import SynaptiqClient

def print_separator():
    print("-" * 60)

def main():
    # Initialisation du client (port par défaut 8000)
    client = SynaptiqClient("http://127.0.0.1:8000")
    
    print("=" * 60)
    print("       🚀 SYNAPTIQ - DEMONSTRATION INTERACTIVE v0 🚀")
    print("=" * 60)
    print("Ce script simule un agent conversationnel doté d'une")
    print("mémoire persistante autonome gérée par SynaptiQ.")
    print("=" * 60)
    
    # 1. Vérification de la connexion à l'API
    print("[1/2] Connexion à l'API SynaptiQ...")
    health = client.health()
    if health.get("status") != "ok":
        print(f"❌ Impossible de se connecter à l'API SynaptiQ ({health.get('error', 'API non démarrée')}).")
        print("\n👉 Assurez-vous d'avoir démarré l'API FastAPI localement sur le port 8000.")
        print("   Exécutez la commande suivante dans un terminal séparé :")
        print("   uvicorn apps.api.main:app --reload --port 8000")
        print("\n👉 Assurez-vous également que le worker tourne :")
        print("   python apps/worker/worker.py")
        sys.exit(1)
        
    print("✅ Connecté avec succès à l'API SynaptiQ.")
    print("Services PostgreSQL (pgvector) & Redis : OK.")
    print_separator()
    
    tenant_id = "demo_org"
    agent_id = "interactive_assistant"
    session_id = "session_interactive_1"
    
    print(f"Initialisation de la session :")
    print(f"  - Tenant : {tenant_id}")
    print(f"  - Agent  : {agent_id}")
    print(f"  - Session: {session_id}")
    print_separator()
    print("Instructions :")
    print("1. Discutez normalement avec l'agent.")
    print("2. Donnez-lui des instructions de style (ex: 'Je préfère que tu tutoies').")
    print("3. Au message suivant, observez comment sa mémoire a été extraite et appliquée.")
    print("Tapez 'exit' ou 'quit' pour quitter.")
    print("=" * 60)
    print("\n[Agent] Bonjour ! Je suis ton assistant SynaptiQ. Comment puis-je t'aider ?")
    
    while True:
        try:
            # Saisie utilisateur
            user_input = input("\n[Vous] > ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print("\n[Agent] Au revoir !")
                break
                
            # --- Étape 2 : Enregistrement de l'événement ---
            print("\n⚙️  Envoi de l'événement à SynaptiQ (asynchrone)...")
            client.capture(tenant_id, agent_id, session_id, user_input)
            
            # Laisser un très court instant pour que le worker traite l'événement en arrière-plan
            # (En conditions réelles, le temps de réponse de l'utilisateur ou du LLM suffit largement)
            print("⚙️  Consolidation de la mémoire en cours...")
            time.sleep(1.5)
            
            # --- Étape 3 : Récupération du contexte mémoire ---
            print("⚙️  Appel à /context/build pour récupérer le contexte mémoire...")
            context_data = client.build_context(
                tenant_id=tenant_id,
                agent_id=agent_id,
                session_id=session_id,
                task="Répondre au message de l'utilisateur",
                query=user_input
            )
            
            context_packet = context_data.get("context_packet", {})
            facts = context_packet.get("facts", [])
            episodes = context_packet.get("episodes", [])
            rules = context_packet.get("rules", [])
            
            # --- Étape 4 : Affichage des souvenirs actifs ---
            print_separator()
            print("🧠 SOUVENIRS REMONTÉS PAR SYNAPTIQ POUR CE MESSAGE :")
            if not facts and not episodes and not rules:
                print("  (Aucune mémoire persistante active pour le moment. L'agent répond à blanc.)")
            else:
                for fact in facts:
                    print(f"  📌 [Sémantique] {fact}")
                for ep in episodes:
                    print(f"  🎬 [Épisodique] {ep}")
                for rule in rules:
                    print(f"  📜 [Procédural] {rule}")
            print_separator()
            
            # --- Étape 5 : Simulation de la réponse re-contextualisée ---
            # On applique les règles extraites pour simuler un comportement LLM
            response_text = "Je prends bien note de ton message."
            
            # Analyser si la mémoire sémantique contient des consignes
            has_pref = False
            for fact in facts:
                if "tutoie" in fact.lower() or "tutoyer" in fact.lower():
                    response_text = "Salut ! Je te réponds en te tutoyant comme tu le souhaites. Qu'est-ce qu'on fait maintenant ?"
                    has_pref = True
                    break
                elif "vouvoie" in fact.lower() or "vouvoyer" in fact.lower():
                    response_text = "Bonjour. Je m'adresse à vous en vous vouvoyant, conformément à vos préférences. Que désirez-vous ?"
                    has_pref = True
                    break
                elif "court" in fact.lower() or "concis" in fact.lower() or "bref" in fact.lower():
                    response_text = "Reçu. Réponse courte."
                    has_pref = True
                    break
            
            if not has_pref:
                # Réponse par défaut amicale
                response_text = f"J'ai bien reçu votre message : '{user_input}'. Ma mémoire SynaptiQ est maintenant à jour !"
                
            print(f"[Agent] {response_text}")
            
        except KeyboardInterrupt:
            print("\n[Agent] Au revoir !")
            break
        except Exception as e:
            print(f"\n❌ Erreur d'interaction : {e}")

if __name__ == "__main__":
    main()
