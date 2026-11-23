"""Client LLM à rotation des modèles gratuits (lot 2). Faux transport HTTP : aucun réseau.

Cas réels du 23/11/2026 rejoués ici : OpenCode Go exige `x-opencode-session` ; un quota
gratuit épuisé répond 429 avec `Retry-After` ; `hy4-flash-free` répond 200 avec un contenu
vide ; la liste des modèles gratuits change d'un jour à l'autre.
"""
import pytest

from synaptiq_core.llm import ClientLLM, LLMIndisponible, depuis_env, extraire_json


class _Rep:
    def __init__(self, status=200, contenu=None, data=None, headers=None, texte=""):
        self.status_code = status
        self._contenu, self._data = contenu, data
        self.headers = headers or {}
        self.text = texte

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        if self._data is not None:
            return self._data
        return {"choices": [{"message": {"content": self._contenu}}]}


class _Horloge:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _client(reponses_par_modele, catalogue=("a-free", "b-free", "c-payant"), **kw):
    appels, entetes = [], []
    horloge = kw.pop("horloge", _Horloge())

    def _get(url, headers=None, timeout=None):
        return _Rep(data={"data": [{"id": m} for m in catalogue]})

    def _post(url, headers=None, json=None, timeout=None):
        appels.append(json["model"])
        entetes.append(headers)
        r = reponses_par_modele.get(json["model"], _Rep(200, "ok"))
        if isinstance(r, Exception):
            raise r
        return r

    client = ClientLLM(base_url="https://opencode.ai/zen/go/v1", api_key="cle",
                       http_get=_get, http_post=_post, horloge=horloge, **kw)
    return client, appels, entetes, horloge


def test_seuls_les_modeles_gratuits_sont_candidats():
    client, _, _, _ = _client({})
    assert client.candidats() == ["a-free", "b-free"]


def test_preference_passe_en_tete_si_encore_proposee():
    client, _, _, _ = _client({}, preference=["b-free", "disparu-free"])
    assert client.candidats() == ["b-free", "a-free"]


def test_429_fait_tourner_et_respecte_retry_after():
    client, appels, _, horloge = _client({"a-free": _Rep(429, headers={"Retry-After": "60"})})
    assert client.chat([{"role": "user", "content": "x"}]).modele == "b-free"
    assert appels == ["a-free", "b-free"]
    assert client.candidats() == ["b-free"]          # a-free en pénalité
    horloge.t += 61
    assert client.candidats() == ["a-free", "b-free"]


def test_reponse_vide_fait_tourner():
    """Cas réel : `hy4-flash-free` répond 200 avec un contenu vide."""
    client, appels, _, _ = _client({"a-free": _Rep(200, "")})
    assert client.chat([{"role": "user", "content": "x"}]).modele == "b-free"


def test_exception_reseau_fait_tourner():
    client, _, _, _ = _client({"a-free": ConnectionError("coupé")})
    assert client.chat([{"role": "user", "content": "x"}]).modele == "b-free"


def test_json_invalide_fait_tourner_en_mode_json():
    client, _, _, _ = _client({"a-free": _Rep(200, "pas du json"),
                               "b-free": _Rep(200, '```json\n{"ok": true}\n```')})
    donnees, modele = client.chat_json([{"role": "user", "content": "x"}])
    assert donnees == {"ok": True} and modele == "b-free"


def test_tous_en_echec_leve_llm_indisponible():
    client, _, _, _ = _client({"a-free": _Rep(500), "b-free": _Rep(503)})
    with pytest.raises(LLMIndisponible):
        client.chat([{"role": "user", "content": "x"}])


def test_entete_de_session_opencode():
    """Sans `x-opencode-session`, OpenCode Go répond 400 MissingSessionID (constaté)."""
    client, _, entetes, _ = _client({})
    client.chat([{"role": "user", "content": "x"}])
    assert entetes[0]["x-opencode-session"] and entetes[0]["Authorization"] == "Bearer cle"


def test_catalogue_rafraichi_et_derniere_liste_gardee_en_cas_de_panne():
    horloge = _Horloge()
    catalogues = [["a-free"], ["a-free", "nouveau-free"]]
    etat = {"panne": False}

    def _get(url, headers=None, timeout=None):
        if etat["panne"]:
            raise ConnectionError("catalogue injoignable")
        return _Rep(data={"data": [{"id": m} for m in catalogues.pop(0)]})

    client = ClientLLM(base_url="http://x/v1", http_get=_get, horloge=horloge,
                       rafraichir_s=100)
    assert client.catalogue() == ["a-free"]
    horloge.t += 50
    assert client.catalogue() == ["a-free"]          # cache
    horloge.t += 60
    assert client.catalogue() == ["a-free", "nouveau-free"]
    etat["panne"] = True
    assert client.catalogue(forcer=True) == ["a-free", "nouveau-free"]


def test_sans_rotation_le_modele_fixe_seul():
    client, appels, _, _ = _client({}, rotation=False, modeles_fixes=["m"])
    client.chat([{"role": "user", "content": "x"}])
    assert appels == ["m"]


@pytest.mark.parametrize("env,attendu", [
    ({"LLM_PROVIDER": "mock"}, None),
    ({"LLM_PROVIDER": "opencode-go"}, None),                       # distant sans clé
    ({"LLM_PROVIDER": "opencode-go", "LLM_API_KEY": "your_api_key"}, None),
])
def test_depuis_env_sans_llm_utilisable(env, attendu):
    assert depuis_env(env) is attendu


def test_depuis_env_opencode_go_tourne_par_defaut():
    client = depuis_env({"LLM_PROVIDER": "opencode-go", "LLM_API_KEY": "k",
                         "LLM_MODEL_PREFERENCE": "beam-free, iceman-free"})
    assert client.rotation and client.base_url == "https://opencode.ai/zen/go/v1"
    assert list(client.preference) == ["beam-free", "iceman-free"]


def test_depuis_env_modele_explicite_desactive_la_rotation():
    client = depuis_env({"LLM_PROVIDER": "opencode-go", "LLM_API_KEY": "k",
                         "LLM_MODEL": "beam-free"})
    assert not client.rotation and list(client.modeles_fixes) == ["beam-free"]


def test_depuis_env_local_sans_cle():
    client = depuis_env({"LLM_PROVIDER": "lmstudio", "LLM_BASE_URL": "http://localhost:1234/v1",
                         "LLM_MODEL": "qwen"})
    assert client is not None and not client.rotation


@pytest.mark.parametrize("texte,attendu", [
    ('{"a": 1}', {"a": 1}),
    ('Voici : {"a": 1} fin', {"a": 1}),
    ('```json\n[1, 2]\n```', [1, 2]),
])
def test_extraire_json(texte, attendu):
    assert extraire_json(texte) == attendu


def test_extraire_json_invalide_leve():
    with pytest.raises(ValueError):
        extraire_json("aucun json ici")


def test_depuis_env_opencode_go_ignore_une_url_locale():
    """Prod du 23/11 : le Compose force LLM_BASE_URL vers le serveur d'embeddings local."""
    client = depuis_env({"LLM_PROVIDER": "opencode-go", "LLM_API_KEY": "k",
                         "LLM_BASE_URL": "http://host.docker.internal:1234/v1"})
    assert client.base_url == "https://opencode.ai/zen/go/v1"
