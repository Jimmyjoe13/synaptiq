# SynaptiQ — démarre les services locaux (API HTTP + serveur MCP en HTTP).
#
# Usage :  .\scripts\start_services.ps1          # démarre ce qui manque
#          .\scripts\start_services.ps1 -Status  # état seulement
#          .\scripts\start_services.ps1 -Stop    # arrête les deux
#
# Idempotent : un service déjà en écoute n'est pas relancé.
#
# ## Pourquoi le serveur MCP tourne en HTTP et non en stdio
#
# En stdio, le client MCP démarre le serveur en processus enfant et l'arrête en fermant
# stdin. Or `mcp.run()` met 141 à 250 ms à se dénouer (boucle anyio de fastmcp), tandis
# qu'antigravity CLI n'accorde qu'environ 100 ms avant d'appeler Kill(). Sur Windows,
# TerminateProcess(handle, 1) se lit « exit status 1 », et son gestionnaire abandonne alors
# le rechargement de TOUS ses serveurs MCP. Les serveurs Node passent sous cette limite,
# Python non — et le délai est dans la bibliothèque, pas dans l'arrêt de l'interpréteur.
#
# En HTTP, il n'y a plus de processus enfant à arrêter : le client se contente d'ouvrir une
# connexion. C'est exactement ce qui a réglé le même problème sur le serveur Obsidian.

param(
    [switch]$Status,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$Racine = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Racine ".venv\Scripts\python.exe"

$Services = @(
    @{ Nom = "API";     Port = 8000; Args = @("-m", "uvicorn", "apps.api.main:app",
                                              "--host", "127.0.0.1", "--port", "8000")
       Journal = "api.log" }
    @{ Nom = "MCP HTTP"; Port = 8765; Args = @((Join-Path $Racine "apps\mcp\server.py"))
       Journal = "mcp.log"
       Env = @{ MCP_TRANSPORT = "http"; MCP_HOST = "127.0.0.1"; MCP_PORT = "8765" } }
)

function Test-Port([int]$Port) {
    $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return [bool]$c
}

function Get-PidSurPort([int]$Port) {
    (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1).OwningProcess
}

if ($Status) {
    foreach ($s in $Services) {
        if (Test-Port $s.Port) {
            Write-Host ("  {0,-9} port {1} : EN ECOUTE (PID {2})" -f $s.Nom, $s.Port, (Get-PidSurPort $s.Port)) -ForegroundColor Green
        } else {
            Write-Host ("  {0,-9} port {1} : arrete" -f $s.Nom, $s.Port) -ForegroundColor Yellow
        }
    }
    exit 0
}

if ($Stop) {
    foreach ($s in $Services) {
        if (Test-Port $s.Port) {
            $procId = Get-PidSurPort $s.Port
            Stop-Process -Id $procId -Force
            Write-Host ("  {0} arrete (PID {1})" -f $s.Nom, $procId) -ForegroundColor Yellow
        }
    }
    exit 0
}

foreach ($s in $Services) {
    if (Test-Port $s.Port) {
        Write-Host ("  {0,-9} deja en ecoute sur {1} (PID {2})" -f $s.Nom, $s.Port, (Get-PidSurPort $s.Port)) -ForegroundColor DarkGray
        continue
    }

    # Les variables d'env du service sont posées pour ce process puis retirées : elles ne
    # doivent pas fuiter sur le service suivant.
    if ($s.Env) { foreach ($k in $s.Env.Keys) { Set-Item -Path ("Env:" + $k) -Value $s.Env[$k] } }

    $sortie = Join-Path $Racine $s.Journal
    $erreurs = Join-Path $Racine ($s.Journal -replace '\.log$', '_error.log')
    Start-Process -FilePath $Py -ArgumentList $s.Args -WorkingDirectory $Racine `
        -WindowStyle Hidden -RedirectStandardOutput $sortie -RedirectStandardError $erreurs

    if ($s.Env) { foreach ($k in $s.Env.Keys) { Remove-Item -Path ("Env:" + $k) -ErrorAction SilentlyContinue } }

    # Attendre l'écoute effective : un « démarré » optimiste ne rend pas service.
    $ok = $false
    foreach ($i in 1..30) {
        Start-Sleep -Milliseconds 400
        if (Test-Port $s.Port) { $ok = $true; break }
    }
    if ($ok) {
        Write-Host ("  {0,-9} demarre sur {1} (PID {2})" -f $s.Nom, $s.Port, (Get-PidSurPort $s.Port)) -ForegroundColor Green
    } else {
        Write-Host ("  {0,-9} N'ECOUTE PAS apres 12 s -> voir {1}" -f $s.Nom, $erreurs) -ForegroundColor Red
    }
}
