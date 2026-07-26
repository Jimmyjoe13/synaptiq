$env:LLM_BASE_URL="http://127.0.0.1:8899/v1"
$env:LLM_API_KEY="dummy"
$env:LLM_MODEL="gpt-oss-120b-medium"
$env:QEM_REDUNDANCY_THRESHOLD="1.0"
$env:QEM_MIN_SCORE_RATIO="0"

Write-Host "=== 🚀 LANCEMENT RUN 1 : RETRIEVAL HYBRIDE (ON) ===" -ForegroundColor Cyan
$env:RETRIEVAL_HYBRID="true"
& .venv\Scripts\python.exe benchmarks/locomo_runner.py benchmarks/locomo10.json --conv 0 --top-k 50 --resume --out benchmarks/final_hybride.json

Write-Host "=== 🚀 LANCEMENT RUN 2 : RETRIEVAL HYBRIDE (OFF) ===" -ForegroundColor Cyan
$env:RETRIEVAL_HYBRID="false"
& .venv\Scripts\python.exe benchmarks/locomo_runner.py benchmarks/locomo10.json --conv 0 --top-k 50 --resume --out benchmarks/final_sans_hybride.json

Write-Host "=== ✅ DEUX RUNS DE BENCHMARK TERMINÉS ===" -ForegroundColor Green
