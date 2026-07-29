# SynaptiQ — équivalent PowerShell du Makefile (Windows, où `make` est absent).
#
# Usage :  .\scripts\dev.ps1 <cible>
# Cibles :  install up down migrate lint types test-unit test-integration test
#           coverage bench bench-explain clean help
#
# Les cibles portent les MÊMES noms que dans le Makefile : une seule documentation
# à maintenir, et un contributeur Windows suit les mêmes instructions que les autres.

param([Parameter(Position = 0)][string]$Cible = "help")

$ErrorActionPreference = "Stop"
$Racine = Split-Path -Parent $PSScriptRoot
Set-Location $Racine

# Le venv jetable du projet s'il existe, sinon le python du PATH.
$Py = Join-Path $Racine ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

if (-not $env:DATABASE_URL) {
    $env:DATABASE_URL = "postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db"
}
if (-not $env:REDIS_URL) { $env:REDIS_URL = "redis://127.0.0.1:6399/0" }

function Invoke-Etape($Titre, [scriptblock]$Bloc) {
    Write-Host "==> $Titre" -ForegroundColor Cyan
    & $Bloc
    if ($LASTEXITCODE -ne 0) { throw "$Titre a échoué (code $LASTEXITCODE)" }
}

switch ($Cible) {
    "install" { Invoke-Etape "Dépendances de développement" { & $Py -m pip install -r requirements-dev.txt } }

    "up" { Invoke-Etape "Infrastructure locale" { docker compose up -d postgres redis } }
    "down" { Invoke-Etape "Arrêt de l'infrastructure" { docker compose down } }

    "migrate" { Invoke-Etape "Migrations Alembic" { & $Py -m alembic upgrade head } }

    "lint" {
        Invoke-Etape "Ruff" {
            & $Py -m ruff check apps packages scripts tests benchmarks migrations examples
        }
    }

    "types" {
        $env:PYTHONPATH = "packages/core"
        Invoke-Etape "Mypy (packages/core)" { & $Py -m mypy }
    }

    "test-unit" {
        Invoke-Etape "Tests unitaires" { & $Py -m pytest tests/unit -q }
    }

    "test-integration" {
        $env:EMBEDDING_PROVIDER = "mock"
        Invoke-Etape "Tests d'intégration" { & $Py -m pytest -m integration -q }
    }

    "test" {
        $env:EMBEDDING_PROVIDER = "mock"
        Invoke-Etape "Suite complète" { & $Py -m pytest -q }
    }

    "coverage" {
        $env:PYTHONPATH = "packages/core"
        Invoke-Etape "Couverture de packages/core" {
            & $Py -m pytest tests/unit -q --cov=synaptiq_core --cov-report=term-missing --cov-fail-under=90
        }
    }

    "bench" {
        # ⚠️ Exige un endpoint LLM joignable (extraction + juge) et prend plusieurs heures
        # sur les 10 conversations. En deçà, l'IC à 95 % vaut ~±8 points et aucun écart de
        # quelques points n'est significatif (cf. synaptiq_core.stats).
        $dataset = if ($env:DATASET) { $env:DATASET } else { "benchmarks/locomo10.json" }
        $conv = if ($env:CONV) { $env:CONV } else { "all" }
        Invoke-Etape "Benchmark LOCOMO ($conv)" {
            & $Py benchmarks/locomo_runner.py $dataset --conv $conv --arm both `
                --qa-workers 4 --out "benchmarks/results_$conv.json"
        }
    }

    "bench-explain" {
        Invoke-Etape "Plans d'exécution du retrieval" {
            & $Py scripts/explain_retrieval.py --seed 5000
        }
    }

    "clean" {
        foreach ($d in ".pytest_cache", ".ruff_cache", ".mypy_cache") {
            if (Test-Path $d) { Remove-Item -Recurse -Force $d }
        }
        if (Test-Path ".coverage") { Remove-Item -Force ".coverage" }
        Get-ChildItem -Recurse -Directory -Filter __pycache__ |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Caches supprimés." -ForegroundColor Green
    }

    default {
        Write-Host "Cibles disponibles :" -ForegroundColor Yellow
        Write-Host "  install           Dépendances de développement"
        Write-Host "  up / down         Infrastructure locale (Postgres + Redis)"
        Write-Host "  migrate           Applique les migrations Alembic"
        Write-Host "  lint / types      Ruff / Mypy"
        Write-Host "  test-unit         Tests sans infrastructure"
        Write-Host "  test-integration  Tests exigeant Postgres + Redis"
        Write-Host "  test              Suite complète"
        Write-Host "  coverage          Couverture de packages/core (seuil CI)"
        Write-Host "  bench             Benchmark LOCOMO (exige un LLM, plusieurs heures)"
        Write-Host "  bench-explain     Vérifie l'usage de l'index HNSW"
        Write-Host "  clean             Supprime les caches"
    }
}
