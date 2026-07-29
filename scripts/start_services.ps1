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
    [switch]$Stop,
    # Secondes d'attente que PostgreSQL et Redis répondent avant de démarrer les services.
    #
    # Indispensable au démarrage de session : Docker Desktop met souvent 1 à 2 minutes à
    # lever ses conteneurs, et une API démarrée avant eux garde un pool NULL et répond 503
    # sur tout — sans jamais se rétablir d'elle-même.
    # 0 pour un lancement manuel quand l'infra est déjà là (le test est de toute façon
    # immédiat si les ports écoutent).
    [int]$WaitForInfra = 0
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

# ─── Attente de l'infrastructure (PostgreSQL + Redis) ───
# Les ports hôte exposés par docker compose. Sans eux, l'API démarre mais reste inutilisable.
$Infra = @( @{ Nom = "PostgreSQL"; Port = 5435 }, @{ Nom = "Redis"; Port = 6399 } )

$manquants = @($Infra | Where-Object { -not (Test-Port $_.Port) })
if ($manquants.Count -gt 0) {
    if ($WaitForInfra -le 0) {
        foreach ($i in $manquants) {
            Write-Host ("  {0,-10} port {1} : INJOIGNABLE" -f $i.Nom, $i.Port) -ForegroundColor Red
        }
        Write-Host "  Demarrer l'infra : docker compose up -d postgres redis" -ForegroundColor Yellow
        Write-Host "  (ou relancer avec -WaitForInfra 300 pour patienter)" -ForegroundColor Yellow
        exit 1
    }
    Write-Host ("  Attente de l'infrastructure (max {0} s)..." -f $WaitForInfra) -ForegroundColor Cyan
    $limite = (Get-Date).AddSeconds($WaitForInfra)
    while ((Get-Date) -lt $limite) {
        if (-not @($Infra | Where-Object { -not (Test-Port $_.Port) })) { break }
        Start-Sleep -Seconds 3
    }
    $encoreManquants = @($Infra | Where-Object { -not (Test-Port $_.Port) })
    if ($encoreManquants.Count -gt 0) {
        foreach ($i in $encoreManquants) {
            Write-Host ("  {0,-10} port {1} : toujours injoignable apres {2} s" -f $i.Nom, $i.Port, $WaitForInfra) -ForegroundColor Red
        }
        exit 1
    }
    Write-Host "  Infrastructure prete." -ForegroundColor Green
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
