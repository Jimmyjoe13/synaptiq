"""Tests unitaires des garde-fous ajoutés le 30/07 après l'audit de l'instance de production.

Deux pannes réelles, deux mécanismes silencieux :

1. Le relais rendait au pool une connexion MORTE après une coupure PostgreSQL, et masquait
   l'erreur d'origine derrière celle du rollback. Il ne se rétablissait plus jamais.
2. Le worker tournait avec un modèle d'embedding différent de celui qui avait écrit les
   vecteurs. Même dimension (384), donc aucune erreur : rappel dégradé en silence.

Aucune infrastructure requise : les deux tests simulent le pool et l'embedder.
"""
import psycopg2
import pytest

# ─── 1. Le relais ne doit pas empoisonner son pool ───────────────────────────

class FauxCurseur:
    def __init__(self, lever: Exception | None = None) -> None:
        self._lever = lever

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, *args, **kwargs):
        if self._lever is not None:
            raise self._lever

    def fetchall(self):
        return []


class FausseConnexion:
    """Connexion dont le SELECT échoue et dont le rollback échoue AUSSI.

    C'est la situation exacte du conteneur `synaptiq-relay` mort le 28/07 : PostgreSQL avait
    coupé, donc `rollback()` levait `InterfaceError: connection already closed`.
    """

    def __init__(self, rollback_echoue: bool) -> None:
        self.rollback_echoue = rollback_echoue
        self.rollback_appele = False

    def cursor(self):
        return FauxCurseur(lever=psycopg2.OperationalError("Connection closed by server."))

    def rollback(self):
        self.rollback_appele = True
        if self.rollback_echoue:
            raise psycopg2.InterfaceError("connection already closed")

    def commit(self):
        pass


class FauxPool:
    def __init__(self, conn) -> None:
        self._conn = conn
        self.rendus: list[tuple[object, bool]] = []

    def getconn(self):
        return self._conn

    def putconn(self, conn, close=False):
        self.rendus.append((conn, close))


def test_relais_ferme_la_connexion_quand_le_rollback_echoue():
    """RÉGRESSION : une connexion morte ne doit JAMAIS retourner dans le pool.

    Sans `close=True`, le pool la redistribue indéfiniment et le relais reste en panne même
    après le retour de PostgreSQL.
    """
    from apps.relay.relay import publish_pending

    conn = FausseConnexion(rollback_echoue=True)
    pool = FauxPool(conn)

    with pytest.raises(psycopg2.OperationalError) as exc:
        publish_pending(pool, redis_client=None)

    # L'erreur remontée est bien celle d'ORIGINE, pas celle du rollback : c'est ce que le
    # traceback de production accusait à tort.
    assert "Connection closed by server" in str(exc.value)
    assert conn.rollback_appele
    assert pool.rendus == [(conn, True)], "la connexion morte doit être fermée, pas recyclée"


def test_relais_recycle_la_connexion_quand_le_rollback_reussit():
    """Un échec SQL ordinaire ne doit pas gaspiller une connexion saine."""
    from apps.relay.relay import publish_pending

    conn = FausseConnexion(rollback_echoue=False)
    pool = FauxPool(conn)

    with pytest.raises(psycopg2.OperationalError):
        publish_pending(pool, redis_client=None)

    assert pool.rendus == [(conn, False)]


def test_relais_recycle_la_connexion_sur_le_chemin_nominal():
    """Le `finally` lit son drapeau sur TOUS les chemins, y compris sans exception."""
    from apps.relay.relay import publish_pending

    class ConnexionSaine(FausseConnexion):
        def cursor(self):
            return FauxCurseur()          # aucune exception : 0 ligne à publier

    conn = ConnexionSaine(rollback_echoue=False)
    pool = FauxPool(conn)

    assert publish_pending(pool, redis_client=None) == 0
    assert pool.rendus == [(conn, False)]


# ─── 2. Le worker refuse un modèle d'embedding incompatible ──────────────────

class _CurseurMemoire:
    def __init__(self, ligne) -> None:
        self._ligne = ligne

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, *args, **kwargs):
        pass

    def fetchone(self):
        return self._ligne


class _ConnMemoire:
    def __init__(self, ligne) -> None:
        self._ligne = ligne

    def cursor(self):
        return _CurseurMemoire(self._ligne)

    def rollback(self):
        pass


class _PoolMemoire:
    def __init__(self, ligne) -> None:
        self._ligne = ligne

    def getconn(self):
        return _ConnMemoire(self._ligne)

    def putconn(self, conn, close=False):
        pass


class _Embedder:
    def __init__(self, vecteur) -> None:
        self._vecteur = vecteur

    def embed_one(self, texte):
        return self._vecteur


def _preparer_worker(monkeypatch, vecteur_stocke, vecteur_recalcule):
    import apps.worker.worker as worker

    ligne = ("un contenu deja stocke", "[" + ",".join(str(v) for v in vecteur_stocke) + "]")
    monkeypatch.setattr(worker, "get_db_pool", lambda: _PoolMemoire(ligne))
    monkeypatch.setattr(worker, "get_embedder", lambda: _Embedder(vecteur_recalcule))
    monkeypatch.setattr(worker, "EMBEDDING_COHERENCE_CHECK", True)
    return worker


def test_worker_accepte_le_meme_modele(monkeypatch):
    """Même modèle -> cosinus 1.000 -> démarrage normal."""
    vecteur = [0.1, 0.2, 0.3, 0.4]
    worker = _preparer_worker(monkeypatch, vecteur, list(vecteur))
    worker.verifier_coherence_embedding()          # ne doit rien lever


def test_worker_refuse_un_modele_different_de_meme_dimension(monkeypatch):
    """LE cas dangereux : dimensions identiques, vecteurs incomparables.

    C'est la panne de l'instance de production — `all-minilm-l6-v2` (anglophone) contre
    `paraphrase-multilingual-minilm-l12-v2`, 384 dimensions tous les deux. Un contrôle de
    dimension laisse passer ; seule la comparaison empirique l'attrape.
    """
    worker = _preparer_worker(monkeypatch, [0.1, 0.2, 0.3, 0.4], [0.4, -0.3, 0.2, -0.1])

    with pytest.raises(SystemExit) as exc:
        worker.verifier_coherence_embedding()
    message = str(exc.value)
    assert "INCOH" in message                      # incohérence signalée...
    assert "cosinus" in message                     # ...avec la mesure qui le prouve
    assert "silence" in message                     # ...et pourquoi rien ne l'aurait dit


def test_worker_refuse_une_dimension_differente(monkeypatch):
    worker = _preparer_worker(monkeypatch, [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3])

    with pytest.raises(SystemExit) as exc:
        worker.verifier_coherence_embedding()
    assert "dimension" in str(exc.value)


def test_worker_laisse_passer_une_base_vierge(monkeypatch):
    """Rien en base : il n'y a rien à contredire, le worker doit démarrer."""
    import apps.worker.worker as worker

    monkeypatch.setattr(worker, "get_db_pool", lambda: _PoolMemoire(None))
    monkeypatch.setattr(worker, "EMBEDDING_COHERENCE_CHECK", True)
    worker.verifier_coherence_embedding()


def test_worker_ne_bloque_pas_sur_une_base_injoignable(monkeypatch):
    """Une base absente au démarrage relève de la boucle principale, pas de ce contrôle."""
    import apps.worker.worker as worker

    def _pool_casse():
        raise psycopg2.OperationalError("could not connect")

    monkeypatch.setattr(worker, "get_db_pool", _pool_casse)
    monkeypatch.setattr(worker, "EMBEDDING_COHERENCE_CHECK", True)
    worker.verifier_coherence_embedding()


def test_worker_controle_desactivable(monkeypatch):
    """Échappatoire explicite : une migration de modèle assumée doit rester possible."""
    worker = _preparer_worker(monkeypatch, [1.0, 0.0], [0.0, 1.0])
    monkeypatch.setattr(worker, "EMBEDDING_COHERENCE_CHECK", False)
    worker.verifier_coherence_embedding()          # aucun SystemExit


# ─── 3. Auto-cicatrisation du pool PostgreSQL et de Redis (API) ───────────────

def test_api_get_db_pool_auto_initialise(monkeypatch):
    """Si db_pool est None (échec au démarrage), get_db_pool l'initialise à la volée."""
    import apps.api.main as main

    pool_cree = FauxPool(None)
    monkeypatch.setattr(main, "db_pool", None)
    monkeypatch.setattr(main.pg_pool, "ThreadedConnectionPool", lambda *a, **k: pool_cree)

    pool = main.get_db_pool()
    assert pool is pool_cree
    assert main.db_pool is pool_cree


def test_api_get_conn_ferme_connexion_quand_rollback_echoue(monkeypatch):
    """Si la connexion PostgreSQL est rompue, get_conn la ferme au lieu de la recycler."""
    import apps.api.main as main

    conn = FausseConnexion(rollback_echoue=True)
    pool = FauxPool(conn)
    monkeypatch.setattr(main, "db_pool", pool)

    with pytest.raises(psycopg2.OperationalError):
        with main.get_conn() as _:
            pass

    assert pool.rendus == [(conn, True)], "la connexion morte doit être fermée (close=True)"


def test_api_get_redis_client_auto_initialise(monkeypatch):
    """Si redis_client est None, get_redis_client l'initialise et le teste avec ping()."""
    import apps.api.main as main

    class FauxRedis:
        def __init__(self):
            self.ping_appele = False
        def ping(self):
            self.ping_appele = True

    faux = FauxRedis()
    monkeypatch.setattr(main, "redis_client", None)
    monkeypatch.setattr(main.redis, "from_url", lambda *a, **k: faux)

    client = main.get_redis_client()
    assert client is faux
    assert faux.ping_appele
    assert main.redis_client is faux
