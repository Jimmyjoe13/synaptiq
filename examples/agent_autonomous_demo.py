import os
import sys
import time

# Ajouter le SDK local au path pour les besoins de la démo
sys.path.append(os.path.join(os.path.dirname(__file__), "../packages/sdk-python"))

from synaptiq_sdk import SynaptiqClient

def print_separator():
    print("-" * 60)

def main():
    client = SynaptiqClient("http://127.0.0.1:8000")
    
    print("=" * 60)
    print("      SYNAPTIQ - APPRENTISSAGE AUTONOME D'AGENT")
    print("=" * 60)
    print("Cette demonstration montre comment un agent IA autonome utilise")
    print("le SDK SynaptiQ comme des OUTILS (Tools) pour lire et ecrire")
    print("de lui-meme dans sa memoire lors de ses cycles de pensee.")
    print("=" * 60)
    
    # Verification connexion
    health = client.health()
    if health.get("status") != "ok":
        print("[-] API SynaptiQ non demarree sur le port 8000.")
        sys.exit(1)
        
    tenant_id = "demo_org"
    agent_id = "autonomous_sales_agent"
    session_id = "session_sales_99"
    
    print("\n--- SIMULATION D'UN CYCLE DE PENSEE DE L'AGENT ---")
    print("Jimmy dit a l'agent :")
    print("[Jimmy] : 'Pour le projet de scraping, limite les appels a 1 par minute pour eviter d'etre bloque.'")
    print_separator()
    time.sleep(1)
    
    # Etape 1 : Thought/Action de l'agent (Ecriture autonome)
    print("[PENSEE DE L'AGENT IA] :")
    print("   \"L'utilisateur me donne une instruction operationnelle importante")
    print("    sur la frequence des appels du scraper. Je dois enregistrer cette")
    print("    regle dans ma memoire procedurale a long terme pour ne pas l'oublier.\"")
    print("\n[ACTION DE L'AGENT IA] : Appel de l'outil 'store_memory()'")
    
    rule_content = "Regle de scraping : limiter le rythme des requetes a 1 appel par minute pour eviter le blocage."
    
    result = client.store_memory(
        tenant_id=tenant_id,
        agent_id=agent_id,
        memory_type="procedural",
        subtype="rule",
        content=rule_content,
        confidence=1.0,
        importance=0.9
    )
    
    print(f"[SYNAPTIQ API RESPONSE] : Memoire sauvegardee avec l'ID : {result.get('memory_id')}")
    print_separator()
    time.sleep(1.5)
    
    print("[Agent] : C'est bien note, Jimmy. J'ai enregistre cette regle de limitation de requetes dans ma memoire.")
    print_separator()
    time.sleep(2)
    
    # Etape 2 : L'utilisateur teste l'agent plus tard ou dans une autre session
    print("\n(Plus tard dans la discussion, Jimmy demande a l'agent...)")
    print("[Jimmy] : 'Quelles sont nos contraintes pour lancer le script de scraping ?'")
    print_separator()
    time.sleep(1)
    
    # Etape 3 : Thought/Action de l'agent (Lecture autonome)
    print("[PENSEE DE L'AGENT IA] :")
    print("   \"L'utilisateur m'interroge sur les contraintes du scraping.")
    print("    Je ne les ai pas dans ma fenetre de contexte immediate. Je dois")
    print("    interroger ma memoire persistante pour voir s'il y a des regles.\"")
    print("\n[ACTION DE L'AGENT IA] : Appel de l'outil 'retrieve(query=\"contraintes scraping\")'")
    
    search_results = client.retrieve(
        tenant_id=tenant_id,
        agent_id=agent_id,
        query="contraintes scraping",
        limit=3,
        memory_type="procedural"
    )
    
    memories = search_results.get("memories", [])
    print("[SYNAPTIQ API RESPONSE] : Souvenirs retrouves :")
    for mem in memories:
        print(f"   -> [{mem['type'].upper()} / {mem['subtype']}] {mem['content']}")
    
    print_separator()
    time.sleep(1.5)
    
    # Etape 4 : Reponse formulee par l'agent a l'aide de ses souvenirs
    print("[PENSEE DE L'AGENT IA] :")
    print("   \"J'ai retrouve la regle. Je formule ma reponse en l'appliquant.\"")
    print("\n[Agent] : Selon ma memoire, nous devons respecter une contrainte :")
    print("          limiter le rythme a 1 appel par minute pour eviter tout blocage.")
    print("=" * 60)
    print("                    FIN DE LA DEMONSTRATION")
    print("=" * 60)

if __name__ == "__main__":
    main()
