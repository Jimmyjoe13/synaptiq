# Benchmarks Q-EM

Exécuter `PYTHONPATH=packages/core python benchmarks/qem_vs_vector.py dataset.jsonl`.
Chaque ligne JSONL définit les identifiants attendus et les candidats scorés par le même retrieval vectoriel. Le script compare le rappel et le budget token du collapse Q-EM au top-k vectoriel de même taille. Les datasets et résultats publiés doivent être versionnés avec le modèle d'embedding et les seuils Q-EM utilisés.
