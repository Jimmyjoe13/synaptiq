# MCP Server — Installation & Operations

The SynaptiQ MCP server (`apps/mcp/server.py`) exposes the memory engine to any MCP client:
Claude Desktop, Cursor, Codex CLI, antigravity CLI, or your own.

It is a **thin HTTP client over the SynaptiQ API** — it never touches PostgreSQL or Redis
itself. That single fact dictates the install order, and most failed setups violate it:

```
docker compose (Postgres + Redis)  →  API on :8000  →  MCP server  →  MCP client
```

If the API is unreachable, every tool call returns an `[ERROR]` string while the server
itself still starts and still lists its tools. **Diagnose from the API outwards, never from
the client inwards.**

---

## 1. Tools

| Tool | Purpose |
|---|---|
| `store_memory(content, memory_type, subtype?)` | Write a memory. Returns which section it was filed into. |
| `recall_memories(query, limit?, memory_type?, collections?)` | Hybrid semantic + full-text search. |
| `build_context(task, query, max_tokens?, collections?)` | Assemble a Q-EM context packet under a token budget. |
| `list_collections()` | The agent's own taxonomy, with volumes, quota and dormant shelves. |
| `create_collection(name, family, description, entangle?)` | Declare a new shelf. |
| `merge_collections(source, target)` | Pour one collection into another; the source is dropped. |

No tool takes `agent_id` or `tenant_id`. Identity is server configuration, never a
parameter — otherwise the model itself would choose whose memory it reads and writes.

---

## 2. Configuration

| Variable | Required | Default | Role |
|---|:---:|---|---|
| `SYNAPTIQ_AGENT_ID` | **yes** | *none* | Memory identity this server reads and writes under. |
| `SYNAPTIQ_API_URL` | no | `http://127.0.0.1:8000` | Base URL of the SynaptiQ API. |
| `SYNAPTIQ_API_KEY` | for `http` | *empty* | Bearer key forwarded to the API. Mandatory when the transport is not `stdio`. |
| `MCP_TRANSPORT` | no | `stdio` | `stdio` or `http`. See §3 — this choice matters. |
| `MCP_HOST` / `MCP_PORT` | no | `0.0.0.0` / `8765` | Listen address in `http` transport. |
| `SYNAPTIQ_AUTOSTART_API` | no | `true` | Spawn `uvicorn` if the API does not answer. Set `false` whenever a supervisor already owns the API. |
| `SYNAPTIQ_AUTOSTART_WAIT_S` | no | `0` | Seconds to wait for the spawned API. `0` = don't block the MCP handshake. |
| `SYNAPTIQ_TIMEOUT_S` | no | `30` | Timeout of API calls. The first call of a session pays for loading the embedding model — measured above 5 s cold. |

> [!IMPORTANT]
> **`SYNAPTIQ_AGENT_ID` has no default, on purpose.**
>
> It once defaulted to a fixed value, and a deployment whose memories had been written under
> a different identity read an empty partition and answered *"no memory found"* — no error,
> no warning. For a memory engine that symptom is indistinguishable from a genuinely empty
> store, so it is undebuggable from the outside.
>
> The server nevertheless **starts** without it and fails at *tool-call* time with a full
> explanation. That is also deliberate: an MCP server that exits during boot only shows the
> client `exit status 1`, its stderr is discarded, and it vanishes from the server list.
> Failing fast is only worth it if somebody reads the failure.
>
> Find the identity your existing memories were written under:
> ```sql
> SELECT agent_id, count(*) FROM memories GROUP BY 1 ORDER BY 2 DESC;
> ```

---

## 3. Choosing a transport

| | `stdio` | `http` |
|---|---|---|
| Process model | Client spawns a child process | Long-lived server, client connects |
| Client entry | `command` + `args` | `serverUrl` |
| Needs `SYNAPTIQ_API_KEY` | no | **yes** |
| Several clients at once | one process each | one server, shared |
| Best for | Claude Desktop, Cursor | Codex CLI, antigravity CLI, containers |

> [!WARNING]
> **Measured limitation of `stdio` with some clients.**
>
> After stdin closes, `mcp.run()` takes **141–250 ms** to unwind. The time is spent inside
> fastmcp's anyio loop, not in interpreter shutdown — an `os._exit()` at the end of
> `mcp.run()` changes nothing (verified).
>
> Clients that grant a shorter grace window (antigravity CLI allows roughly 100 ms) call
> `Kill()` first. On Windows `TerminateProcess(handle, 1)` reads as `exit status 1`, and such
> a client may then abandon the reload of **every** MCP server it manages. Node servers fit
> under that window; Python does not.
>
> With those clients, use `http`: there is no child process to stop, so no grace window to
> respect.

---

## 4. Install — Docker

The MCP service sits behind a Compose profile, so it is **not** started by a plain
`docker compose up`:

```bash
cp .env.example .env
# set SYNAPTIQ_AGENT_ID and SYNAPTIQ_API_KEY in .env first
docker compose --profile mcp-http up -d
```

It then listens on `127.0.0.1:8765`, with `SYNAPTIQ_API_URL=http://api:8000` on the internal
network. Declare it client-side with `serverUrl` (§6).

---

## 5. Install — host process (no container for Python)

Use this when embeddings come from a local model server (LM Studio, Ollama) that only listens
on loopback, since a container cannot reach it.

**1. Clone and create a virtualenv (Python 3.11+)**

```bash
git clone https://github.com/Jimmyjoe13/synaptiq.git
cd synaptiq
python -m venv .venv
# fastmcp MUST be installed in the SAME command as requirements.txt. Installed separately,
# the resolver bumps starlette past what fastapi==0.115.6 supports, and the API dies with:
#   TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'
.venv/bin/pip install -r requirements.txt fastmcp
```

**2. Write `.env`**

```env
DATABASE_URL=postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db
REDIS_URL=redis://127.0.0.1:6399/0

EMBEDDING_PROVIDER=lmstudio
EMBEDDING_BASE_URL=http://localhost:1234/v1
EMBEDDING_MODEL=text-embedding-paraphrase-multilingual-minilm-l12-v2.gguf
EMBEDDING_DIM=384

SYNAPTIQ_TENANT=default
SYNAPTIQ_AUTH_REQUIRED=true

SYNAPTIQ_API_URL=http://127.0.0.1:8000
SYNAPTIQ_AGENT_ID=my_agent
SYNAPTIQ_API_KEY=sk-synaptiq-...
MCP_TRANSPORT=http
MCP_HOST=127.0.0.1
MCP_PORT=8765
```

> [!CAUTION]
> **On Windows, keep `.env` pure ASCII.** `slowapi` re-reads it through
> `starlette.config.Config`, which opens it *without* specifying an encoding — so cp1252. One
> non-representable UTF-8 byte (an emoji is enough: `0x8f`, from the U+FE0F variation
> selector) crashes the API at boot on an opaque `UnicodeDecodeError`, far from its cause.
>
> **Never change `EMBEDDING_MODEL` on a populated instance without re-embedding.** Two
> different 384-dim models raise no error at all — the stored vectors simply stop being
> comparable and recall degrades *in silence*. The worker refuses to start on a mismatch
> (`EMBEDDING_COHERENCE_CHECK`); verify manually with the cosine between a stored vector and
> the one recomputed by the current model — it must be `1.000`.

**3. Issue an API key with the minimum scope**

`read` + `write`, restricted to that one agent, and **no `admin`** — so the GDPR purge stays
out of the model's reach:

```bash
.venv/bin/python scripts/create_api_key.py --name "mcp-my-agent" \
    --scopes read write --agents my_agent
```

Copy the printed key into `SYNAPTIQ_API_KEY`. Only its SHA-256 hash is stored; it is never
recoverable.

**4. Start storage, then the services**

```bash
docker compose up -d postgres redis migrate
python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 &
python apps/mcp/server.py &
```

<details>
<summary><strong>Windows: <code>scripts/start_services.ps1</code> and a logon task</strong></summary>

<br>

```powershell
.\scripts\start_services.ps1 -WaitForInfra 300     # start API + MCP
.\scripts\start_services.ps1 -Status               # what is listening
.\scripts\start_services.ps1 -Stop                 # stop both
```

The script is idempotent — a port already listening is left alone. `-WaitForInfra 300` is what
makes it safe at logon: Docker Desktop often needs one to two minutes to raise its containers,
and an API started before them keeps a NULL connection pool and answers `503` to everything,
**forever** — it never recovers on its own.

```powershell
$repo    = "C:\path\to\synaptiq"
$action  = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -File `"$repo\scripts\start_services.ps1`" -WaitForInfra 300"
$trigger = New-ScheduledTaskTrigger -AtLogOn
Register-ScheduledTask -TaskName "SynaptiQ services" -Action $action -Trigger $trigger
```

</details>

---

## 6. Declaring the server client-side

### `http` transport

The whole entry is one line — there is no process for the client to manage.

```json
{
  "mcpServers": {
    "synaptiq": { "serverUrl": "http://127.0.0.1:8765/mcp/" }
  }
}
```

Config file location depends on the client — for example `~/.codex/config.toml` for Codex
CLI (`[mcp_servers.synaptiq]` / `url = "..."`), or
`~/.gemini/antigravity-cli/mcp_config.json` for antigravity CLI.

### `stdio` transport

```json
{
  "mcpServers": {
    "synaptiq": {
      "command": "/absolute/path/to/synaptiq/.venv/bin/python",
      "args": ["/absolute/path/to/synaptiq/apps/mcp/server.py"],
      "env": {
        "SYNAPTIQ_API_URL": "http://127.0.0.1:8000",
        "SYNAPTIQ_API_KEY": "sk-synaptiq-...",
        "SYNAPTIQ_AGENT_ID": "my_agent",
        "MCP_TRANSPORT": "stdio",
        "SYNAPTIQ_AUTOSTART_API": "false"
      }
    }
  }
}
```

See [`examples/claude_desktop_config.json`](../examples/claude_desktop_config.json). Two rules:

- **Absolute paths** to both the interpreter and `server.py` — not `-m apps.mcp.server` with a
  `cwd`. The script fixes up `sys.path` itself and runs from any working directory; clients
  that ignore `cwd` would otherwise fail with `ModuleNotFoundError`, which surfaces only as an
  opaque `exit status 1`.
- **`SYNAPTIQ_AUTOSTART_API=false`** when a supervisor already owns the API, otherwise the MCP
  server spawns a second `uvicorn` that loses the race on port 8000.

---

## 7. Verifying the install

Run all three. The first two can pass while the memory is still silently empty.

```bash
# 1. The port is listening
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8765/mcp/    # 406 = alive, awaiting a handshake

# 2. The API is actually healthy — not merely bound to its port
curl http://127.0.0.1:8000/v1/health
# -> {"status":"ok","services":{"postgres":"healthy","redis":"healthy","ingestion":"healthy"}}

# 3. THE ONE THAT MATTERS: the configured identity owns the memories
docker exec synaptiq-postgres psql -U synaptiq -d synaptiq_db \
  -c "SELECT agent_id, count(*) FROM memories GROUP BY 1 ORDER BY 2 DESC;"
```

`SYNAPTIQ_AGENT_ID` must appear in that list with a non-zero count. A mismatch is the one
failure the tools cannot report: `recall_memories` answers *"no matching memory found"*, which
reads exactly like a fresh install.

Then, from the client, ask the agent to recall something you know is stored.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Client shows `exit status 1`, all MCP servers stop reloading | `stdio` shutdown exceeds the client's grace window (§3) | Switch to `MCP_TRANSPORT=http`, declare `serverUrl` |
| `[ERROR] ... SYNAPTIQ_AGENT_ID n'est pas defini` | No identity configured | Set `SYNAPTIQ_AGENT_ID` in the server env |
| *"No matching memory found"*, no error | Identity mismatch, **or** `EMBEDDING_MODEL` differs from the one that wrote the vectors | Check §7 step 3; the cosine of a stored vector vs. a recomputed one must be `1.000` |
| Tool call fails with `Connection refused` | API down, or wrong `SYNAPTIQ_API_URL` | `curl /v1/health`; read `api_error.log` |
| Tool call fails with `Read timed out` on the first call | Cold model load exceeds `SYNAPTIQ_TIMEOUT_S` | Warm the model, or raise `SYNAPTIQ_TIMEOUT_S` |
| `401 Clé API requise` on every tool | `SYNAPTIQ_AUTH_REQUIRED=true` and no `SYNAPTIQ_API_KEY` in the MCP env | Issue a key (§5 step 3) and set it |
| `403 Permission 'write' absente` | Key issued read-only | Re-issue with `--scopes read write` |
| `[INFO] '<x>' n'est pas une collection declaree` | Writing to an undeclared shelf; the memory landed in the family's fallback section | `create_collection(...)`, or reuse an existing one |
| API dies at boot on `UnicodeDecodeError` | Non-ASCII byte in `.env` on Windows | Rewrite `.env` in pure ASCII |
| `TypeError: Router.__init__() ... 'on_startup'` | `fastmcp` installed separately from `requirements.txt` | Reinstall both in one `pip install` |
| Tools work, but `/events` never becomes a memory | `relay` and/or `worker` not running | `docker compose up -d relay worker`. `/events` returns `201` and queues in the outbox even when nothing consumes it — `/v1/health` reports `"ingestion":"stalled"` |

The MCP server logs to **stderr**, never stdout — in `stdio` transport stdout carries the
JSON-RPC frames, and a single log line there corrupts the session.

> [!IMPORTANT]
> **A breakdown and a misconfiguration do not exit through the same door.**
>
> A **breakdown** — timeout, API unreachable, 5xx — raises a `ToolError`, so the client sees
> `isError: true`. These tools used to return `"[ERROR] ..."` as ordinary text for every
> failure, and to a client that string is a perfectly valid *result*: a `Read timed out` once
> surfaced in an agent's conversation as though it were the content of the memory. For a
> memory engine that confusion is the worst possible one — a silent failure is
> indistinguishable from an empty memory.
>
> A **misconfiguration** — `SYNAPTIQ_AGENT_ID` missing — stays plain text on purpose. Nothing
> is broken; the message carries the fix, and the agent can relay it verbatim.
