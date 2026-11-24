"""OpenRouter backend for DarkGPT.

Chat now goes straight to OpenRouter's OpenAI-compatible API instead of g4f.
The retired g4f providers this app shipped with (DDG, Blackbox, MetaAI,
HuggingFace) all returned 404s / dead endpoints in 2026, so every chat failed.

Only zero-cost models are offered: models whose prompt AND completion price is
0 are fetched live from OpenRouter's public /models endpoint (no key needed for
that call) and cached for an hour. Nothing is hardcoded, because the free tier
rotates constantly.

The API key is read from, in order:
  1. an explicit `api_key` argument (the Streamlit sidebar), or
  2. the OPENROUTER_API_KEY environment variable, or
  3. a .env file in the project root (OPENROUTER_API_KEY=sk-or-...)

A free key is created at https://openrouter.ai/keys - no card required.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Iterator, List, Optional
from urllib.parse import urlparse

import requests

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODELS_URL = "https://integrate.api.nvidia.com/v1/models"

# Providers DarkGPT can talk to. "nvidia" uses NVIDIA NIM directly with the
# user's own NVIDIA_API_KEY, which is not subject to OpenRouter's free daily cap.
PROVIDERS: Dict[str, Dict[str, str]] = {
    "openrouter": {
        "label": "OpenRouter",
        "chat_url": OPENROUTER_CHAT_URL,
        "models_url": OPENROUTER_MODELS_URL,
        "env_key": "OPENROUTER_API_KEY",
        "default_model": "openrouter/free",
        "signup": "https://openrouter.ai/keys",
    },
    "nvidia": {
        "label": "NVIDIA NIM",
        "chat_url": NVIDIA_CHAT_URL,
        "models_url": NVIDIA_MODELS_URL,
        "env_key": "NVIDIA_API_KEY",
        "default_model": "nvidia/nemotron-3-ultra-550b-a55b",
        "signup": "https://build.nvidia.com/settings/api-keys",
    },
}


def current_provider() -> str:
    """Configured chat provider, falling back safely to OpenRouter."""
    try:
        from utils import appconfig

        name = str(appconfig.get("provider") or "openrouter").strip().lower()
    except Exception:
        name = "openrouter"
    return name if name in PROVIDERS else "openrouter"


def provider_config(name: Optional[str] = None) -> Dict[str, str]:
    return PROVIDERS[(name or current_provider()).lower()]


def chat_url(name: Optional[str] = None) -> str:
    return provider_config(name)["chat_url"]


def models_url(name: Optional[str] = None) -> str:
    return provider_config(name)["models_url"]


def _provider_label() -> str:
    """Display name of the backend that will actually receive the request."""
    return str(provider_config()["label"] or "the API")


MAX_RETRY_AFTER = 20.0  # never block the TUI longer than this on a 429


def _retry_after_seconds(resp, default: float) -> float:
    """Read ``Retry-After`` (seconds or HTTP date) and cap it."""
    raw = ""
    try:
        raw = str(resp.headers.get("Retry-After") or "").strip()
    except Exception:
        raw = ""
    if raw:
        try:
            return max(0.0, min(float(raw), MAX_RETRY_AFTER))
        except (TypeError, ValueError):
            try:
                from email.utils import parsedate_to_datetime

                when = parsedate_to_datetime(raw)
                delta = when.timestamp() - time.time()
                return max(0.0, min(delta, MAX_RETRY_AFTER))
            except Exception:
                pass
    return max(0.0, min(default, MAX_RETRY_AFTER))


def _is_daily_limit(body: str) -> bool:
    """True for OpenRouter's per-day free-tier cap (never helped by retries)."""
    lowered = (body or "").lower()
    markers = (
        "free-models-per-day",
        "openrouter_free_tier_daily",
        "free model requests per day",
        "daily limit",
    )
    return any(marker in lowered for marker in markers)


def _backoff_delay(attempt: int, base: float) -> float:
    """Exponential backoff with jitter, capped so the UI stays responsive."""
    import random

    delay = base * (2 ** attempt)
    return min(8.0, delay) * (0.7 + random.random() * 0.6)


def post_with_retry(
    url: str,
    *,
    headers: Dict[str, str],
    payload: Dict,
    timeout,
    stream: bool = False,
    attempts: int = 3,
    backoff: float = 0.8,
):
    """POST with bounded exponential backoff + jitter for transient failures.

    DNS resolution is left to ``requests`` and retried like any other
    connection error. A separate 5s resolver preflight used to hard-fail on a
    momentary resolver stall even though the real request would have succeeded,
    so it was removed. ``requests`` applies ``timeout`` to the whole connect
    (DNS included), and the bounded retries keep the TUI responsive.

    Streaming is safe because retries only happen before a successful response
    object is returned.

    HTTP 402 is deterministic (budget/in-flight) and returned immediately.
    HTTP 429 is retried while honouring ``Retry-After`` (capped).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        last_attempt = attempt >= attempts - 1
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout, stream=stream)
            if resp.status_code == 402:
                # Budget / in-flight failures are deterministic: never retry.
                return resp
            if resp.status_code == 429 and not last_attempt:
                detail = ""
                try:
                    detail = resp.text[:2000]
                except Exception:
                    pass
                if _is_daily_limit(detail):
                    # A daily cap will not reset because we retried. Each retry
                    # is another billable-against-quota request, so stop now.
                    return resp
                wait = _retry_after_seconds(resp, _backoff_delay(attempt, backoff))
                resp.close()
                time.sleep(wait)
                continue
            if resp.status_code in (502, 503, 504) and not last_attempt:
                resp.close()
                time.sleep(_backoff_delay(attempt, backoff))
                continue
            return resp
        except requests.ConnectionError as exc:
            last_exc = exc
            if not last_attempt:
                time.sleep(_backoff_delay(attempt, backoff))
                continue
            raise
        except requests.Timeout as exc:
            last_exc = exc
            if not last_attempt:
                time.sleep(_backoff_delay(attempt, backoff))
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise OpenRouterError(f"{_provider_label()} request failed after retries.")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(PROJECT_ROOT, ".cache", "free_models.json")
NVIDIA_CACHE_PATH = os.path.join(PROJECT_ROOT, ".cache", "nvidia_models.json")
CACHE_TTL_SECONDS = 3600
CACHE_VERSION = 2  # bump when the cached entry shape changes

# Used only if the live lookup fails (offline / rate limited). Auto-router first.
FALLBACK_FREE_MODELS = [
    "openrouter/free",
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
]

# Used when NVIDIA's model catalogue is unreachable but the key is configured.
FALLBACK_NVIDIA_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "google/gemma-4-31b-it",
]

# NVIDIA's /v1/models response carries no capability metadata, so filter out
# the embedding / guardrail / CLIP endpoints that cannot serve chat completions.
# Showing those in /model would produce a 404 like "bigcode/starcoder2-15b" did.
NVIDIA_NON_CHAT_MARKERS = (
    "embed", "clip", "guard", "safety", "retriever", "detector",
    "calibration", "riva-translate", "gliner", "starcoder", "synthetic-video",
)

# Models whose own training leaves out the refusal layer (Dolphin / Venice /
# TheDrummer / Hermes lineage). These are uncensored *by design* - picking one is
# a product choice, not a jailbreak. None of them are free, and OpenRouter has no
# free uncensored model at all, so they stay hidden unless the user opts in.
UNCENSORED_MODEL_IDS = [
    "cognitivecomputations/dolphin-mistral-24b-venice-edition",
    "thedrummer/cydonia-24b-v4.1",
    "thedrummer/unslopnemo-12b",
    "thedrummer/skyfall-36b-v2",
    "sao10k/l3.1-euryale-70b",
    "gryphe/mythomax-l2-13b",
]

APP_TITLE = "DarkGPT"
APP_REFERER = "http://127.0.0.1:8501"


class OpenRouterError(RuntimeError):
    """Raised with a message that is safe and useful to show in the UI.

    Structured provider errors (notably HTTP 402 budget/in-flight failures) keep
    their machine-readable fields so the agent and TUI can react deterministically
    instead of retrying or looping.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        code: Optional[int] = None,
        reason: str = "",
        limit_source: str = "",
        remedy_hint: str = "",
        retry_after: Optional[int] = None,
        provider_name: str = "",
        body: str = "",
        reset_at: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = int(status_code or 0)
        self.code = code
        self.reason = reason or ""
        self.limit_source = limit_source or ""
        self.remedy_hint = remedy_hint or ""
        self.retry_after = retry_after
        self.provider_name = provider_name or ""
        self.body = body or ""
        self.reset_at = reset_at

    @property
    def is_daily_limit(self) -> bool:
        return self.is_rate_limit and (
            "free-models-per-day" in self.message.lower()
            or "openrouter_free_tier_daily" in self.limit_source.lower()
            or "free model requests per day" in self.message.lower()
        )

    @property
    def is_budget(self) -> bool:
        return (
            self.status_code == 402
            or self.code == 402
            or self.reason in ("in_flight_budget_exhausted", "insufficient_credits")
        )

    @property
    def is_rate_limit(self) -> bool:
        return self.status_code == 429 or self.code == 429 or self.reason == "rate_limited"

    def card(self) -> Dict[str, object]:
        reason = self.reason
        if not reason:
            if self.is_budget:
                reason = "in_flight_budget_exhausted"
            elif self.is_rate_limit:
                reason = "rate_limited"
            elif self.status_code >= 500:
                reason = "provider_unavailable"
            elif self.status_code == 401:
                reason = "unauthorized"
            else:
                reason = f"http_{self.status_code or 'error'}"
        return {
            "kind": "budget" if self.is_budget else ("rate_limit" if self.is_rate_limit else "http"),
            "status_code": self.status_code or self.code or 0,
            "reason": reason,
            "limit_source": self.limit_source,
            "message": self.message,
            "remedy_hint": self.remedy_hint,
            "retry_after": self.retry_after,
            "provider_name": self.provider_name,
            "reset_at": self.reset_at,
            "daily_limit": self.is_daily_limit,
        }


def _coerce_retry_after(value: object) -> Optional[int]:
    try:
        return max(0, int(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coerce_reset_at(value: object) -> Optional[int]:
    """Normalise X-RateLimit-Reset (epoch seconds/ms, delta, or HTTP date)."""
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        number = float(raw)
    except (TypeError, ValueError):
        try:
            from email.utils import parsedate_to_datetime

            return int(parsedate_to_datetime(raw).timestamp())
        except Exception:
            return None
    if number > 1e12:  # epoch milliseconds
        return int(number / 1000)
    if number > 1e9:  # epoch seconds
        return int(number)
    return int(time.time() + max(0, number))  # delta seconds


def parse_error_response(response: object, body: Optional[str] = None) -> OpenRouterError:
    """Turn a non-200 OpenRouter response into a structured error.

    Reads both the JSON body (``error.metadata.*``) and the ``Retry-After``
    header, because 402 responses put the useful values in both places.
    """
    status = int(getattr(response, "status_code", 0) or 0)
    text = body if body is not None else str(getattr(response, "text", "") or "")
    payload: Dict[str, object] = {}
    try:
        raw = json.loads(text)
        if isinstance(raw, dict):
            err = raw.get("error")
            payload = err if isinstance(err, dict) else raw
    except Exception:
        payload = {}

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    headers = metadata.get("headers") if isinstance(metadata.get("headers"), dict) else {}
    header_retry: object = None
    try:
        header_retry = response.headers.get("Retry-After")  # type: ignore[attr-defined]
    except Exception:
        header_retry = None

    retry_after = _coerce_retry_after(
        headers.get("Retry-After")
        or headers.get("retry-after")
        or header_retry
    )
    reset_header: object = None
    remaining_header: object = None
    try:
        reset_header = response.headers.get("X-RateLimit-Reset")  # type: ignore[attr-defined]
        remaining_header = response.headers.get("X-RateLimit-Remaining")  # type: ignore[attr-defined]
    except Exception:
        pass
    reset_at = _coerce_reset_at(
        headers.get("X-RateLimit-Reset") or reset_header
        or metadata.get("reset_at") or payload.get("reset_at")
    )
    if reset_at is not None and retry_after is None:
        retry_after = _coerce_retry_after(max(0, reset_at - time.time()))
    message = str(payload.get("message") or text[:400] or f"HTTP {status}")
    code: Optional[int]
    try:
        raw_code = payload.get("code")
        code = int(raw_code) if raw_code is not None else None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        code = None

    return OpenRouterError(
        message,
        status_code=status,
        code=code,
        reason=str(metadata.get("reason") or payload.get("reason") or ""),
        limit_source=str(metadata.get("limit_source") or payload.get("limit_source") or ""),
        remedy_hint=str(metadata.get("remedy_hint") or payload.get("remedy_hint") or ""),
        retry_after=retry_after,
        provider_name=str(metadata.get("provider_name") or ""),
        body=text[:2000],
        reset_at=reset_at,
    )


# --------------------------------------------------------------------------
# API key handling
# --------------------------------------------------------------------------
def _read_dotenv(path: Optional[str] = None) -> Dict[str, str]:
    """Minimal .env parser: KEY=VALUE, ignores comments and blank lines."""
    path = path or os.path.join(PROJECT_ROOT, ".env")
    values: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                values[key.strip()] = val.strip().strip("'\"")
    except OSError:
        pass
    return values


def get_api_key(explicit: Optional[str] = None, provider: Optional[str] = None) -> str:
    """Resolve the active provider's key from argument, environment, or .env."""
    if explicit and explicit.strip():
        return explicit.strip()
    cfg = provider_config(provider)
    env_key = os.getenv(cfg["env_key"], "").strip()
    if env_key:
        return env_key
    return _read_dotenv().get(cfg["env_key"], "").strip()


def auth_headers(explicit: Optional[str] = None, provider: Optional[str] = None) -> Dict[str, str]:
    """Build OpenAI-compatible auth headers for the active provider."""
    cfg = provider_config(provider)
    key = get_api_key(explicit, provider)
    if not key:
        raise OpenRouterError(
            f"No API key configured for {cfg['label']}. Add "
            f"{cfg['env_key']}=... to {os.path.join(PROJECT_ROOT, '.env')} "
            f"(get a key at {cfg['signup']})."
        )
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if provider_config(provider)["label"] == "OpenRouter":
        headers["HTTP-Referer"] = APP_REFERER
        headers["X-Title"] = APP_TITLE
    return headers


# --------------------------------------------------------------------------
# Free model discovery
# --------------------------------------------------------------------------
def _is_free(model: Dict) -> bool:
    pricing = model.get("pricing") or {}
    try:
        prompt = float(pricing.get("prompt") or 0)
        completion = float(pricing.get("completion") or 0)
    except (TypeError, ValueError):
        return False
    return prompt == 0.0 and completion == 0.0


def _is_text_chat(model: Dict) -> bool:
    modality = ((model.get("architecture") or {}).get("modality") or "").lower()
    return modality.endswith("->text") and "audio" not in modality


def _fetch_catalog() -> Dict[str, List[Dict]]:
    """Fetch OpenRouter's catalogue once and split it into free / uncensored."""
    resp = requests.get(OPENROUTER_MODELS_URL, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    free: List[Dict] = []
    uncensored: List[Dict] = []
    for model in payload.get("data", []):
        model_id = model.get("id", "")
        modality = ((model.get("architecture") or {}).get("modality") or "").lower()
        if model_id in UNCENSORED_MODEL_IDS:
            pricing = model.get("pricing") or {}
            try:
                price_in = float(pricing.get("prompt") or 0) * 1_000_000
                price_out = float(pricing.get("completion") or 0) * 1_000_000
            except (TypeError, ValueError):
                price_in = price_out = 0.0
            uncensored.append(
                {
                    "id": model_id,
                    "name": model.get("name") or model_id,
                    "context": int(model.get("context_length") or 0),
                    "price_in": round(price_in, 3),
                    "price_out": round(price_out, 3),
                    "uncensored": True,
                    "free": False,
                    "vision": "image" in modality,
                }
            )
            continue

        if not _is_free(model) or not _is_text_chat(model):
            continue
        free.append(
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "context": int(model.get("context_length") or 0),
                "price_in": 0.0,
                "price_out": 0.0,
                "uncensored": False,
                "free": True,
                "vision": "image" in modality,
            }
        )

    free.sort(key=lambda m: (m["id"] != "openrouter/free", -m["context"]))
    uncensored.sort(key=lambda m: m["price_in"])
    return {"free": free, "uncensored": uncensored}


def _read_cache(path: Optional[str] = None) -> Optional[Dict[str, List[Dict]]]:
    try:
        with open(path or CACHE_PATH, "r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if cached.get("version") != CACHE_VERSION:
            return None
        if time.time() - cached.get("fetched_at", 0) < CACHE_TTL_SECONDS:
            models = cached.get("models") or []
            if models:
                # Older cache format stored a bare list of free models: refetch so
                # the uncensored set is present too.
                if isinstance(models, list):
                    return None
                return models
    except (OSError, ValueError, TypeError):
        pass
    return None


def _write_cache(catalog: Dict[str, List[Dict]], path: Optional[str] = None) -> None:
    try:
        target = path or CACHE_PATH
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump({"version": CACHE_VERSION, "fetched_at": time.time(), "models": catalog}, fh)
    except OSError:
        pass


def _fallback_catalog() -> Dict[str, List[Dict]]:
    return {
        "free": [
            {"id": m, "name": m, "context": 0, "price_in": 0.0, "price_out": 0.0,
             "uncensored": False, "free": True, "vision": False}
            for m in FALLBACK_FREE_MODELS
        ],
        "uncensored": [],
    }


def _fetch_nvidia_catalog() -> Dict[str, List[Dict]]:
    """Fetch NVIDIA NIM's model catalogue; every model is usable with your key."""
    resp = requests.get(NVIDIA_MODELS_URL, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data", []) if isinstance(payload, dict) else []
    free: List[Dict] = []
    for model in data if isinstance(data, list) else []:
        model_id = str(model.get("id") or "").strip()
        if not model_id:
            continue
        if any(marker in model_id.lower() for marker in NVIDIA_NON_CHAT_MARKERS):
            continue
        modality = str(model.get("modality") or "").lower()
        free.append({
            "id": model_id,
            "name": str(model.get("name") or model_id),
            "context": int(model.get("context_length") or model.get("max_context") or 0),
            "price_in": 0.0,
            "price_out": 0.0,
            "uncensored": False,
            "free": True,
            "vision": "image" in modality,
        })
    if not free:
        raise ValueError("NVIDIA catalogue was empty")
    free.sort(key=lambda m: m["id"])
    return {"free": free, "uncensored": []}


def _fallback_nvidia_catalog() -> Dict[str, List[Dict]]:
    return {
        "free": [
            {"id": m, "name": m, "context": 0, "price_in": 0.0, "price_out": 0.0,
             "uncensored": False, "free": True, "vision": False}
            for m in FALLBACK_NVIDIA_MODELS
        ],
        "uncensored": [],
    }


def get_catalog(force_refresh: bool = False, provider: Optional[str] = None) -> Dict[str, List[Dict]]:
    """Model catalogue for the active provider, cached for one hour."""
    name = (provider or current_provider()).lower()
    if name == "nvidia":
        if not force_refresh:
            cached = _read_cache(NVIDIA_CACHE_PATH)
            if cached:
                return cached
        try:
            catalog = _fetch_nvidia_catalog()
            _write_cache(catalog, NVIDIA_CACHE_PATH)
            return catalog
        except (requests.RequestException, ValueError, TypeError):
            return _fallback_nvidia_catalog()

    if not force_refresh:
        cached = _read_cache()
        if cached:
            return cached
    try:
        catalog = _fetch_catalog()
        if catalog.get("free"):
            _write_cache(catalog)
            return catalog
    except (requests.RequestException, ValueError):
        pass
    return _fallback_catalog()


def list_free_models(force_refresh: bool = False, provider: Optional[str] = None) -> List[Dict]:
    """Return text chat models for the active provider."""
    return get_catalog(force_refresh=force_refresh, provider=provider)["free"]


def list_models(include_uncensored: bool = False) -> List[Dict]:
    """Free models, plus the paid uncensored-by-design set when opted in."""
    catalog = get_catalog()
    models = list(catalog["free"])
    if include_uncensored:
        models.extend(catalog.get("uncensored") or [])
    return models


def _label(model: Dict) -> str:
    ctx = model["context"]
    short = model["id"]
    suffix = ""
    if model.get("uncensored"):
        suffix = f"  [uncensored ${model['price_in']:.2f}/${model['price_out']:.2f} per M]"
    elif ctx:
        ctx_txt = f"{ctx // 1000}k" if ctx < 1_000_000 else f"{ctx // 1_000_000}M"
        suffix = f"  ({ctx_txt} ctx)"
    if model.get("vision"):
        suffix += "  ? images"
    return f"{short}{suffix}"


def get_model_info(model_id: str) -> Optional[Dict]:
    """Catalogue entry for a model id, or None if it is not in the free/uncensored sets."""
    catalog = get_catalog()
    for group in ("free", "uncensored"):
        for entry in catalog.get(group) or []:
            if entry.get("id") == model_id:
                return entry
    return None


def supports_images(model_id: str) -> bool:
    """True when the model advertises image input."""
    info = get_model_info(model_id)
    if info is not None:
        return bool(info.get("vision"))
    return False


def build_model_options(models: List[Dict]) -> Dict[str, str]:
    """Dropdown label -> OpenRouter model id."""
    return {_label(m): m["id"] for m in models}


# Display name -> OpenRouter model id. Free models only, for backwards compatibility.
display_model_mapping: Dict[str, str] = build_model_options(list_free_models())



def get_model(display_model: str) -> str:
    """Map a dropdown label to an OpenRouter model id."""
    return display_model_mapping.get(display_model, display_model)


def get_provider(model: str) -> str:
    """Kept for backwards compatibility; OpenRouter routes providers itself."""
    return "OpenRouter"


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------
def _explain_http_error(status: int, body: str) -> str:
    """Explain an HTTP failure in terms of the provider that returned it."""
    body = (body or "").strip()
    cfg = provider_config()
    label = cfg["label"]
    is_openrouter = cfg["env_key"] == "OPENROUTER_API_KEY"
    if status == 401:
        if is_openrouter:
            return (
                "OpenRouter rejected the API key (401). Create a free key at "
                "https://openrouter.ai/keys and paste it in the sidebar."
            )
        return (
            f"{label} rejected the API key (401). Check {cfg['env_key']} in "
            f"{os.path.join(PROJECT_ROOT, '.env')} or regenerate it at {cfg['signup']}."
        )
    if status == 402:
        if is_openrouter:
            return "OpenRouter says this account has insufficient credit (402). Pick a ':free' model."
        return f"{label} says this account has insufficient credit (402). Check your NVIDIA trial credits."
    if status == 429:
        if is_openrouter:
            return (
                "Rate limited by OpenRouter (429). Free models allow roughly 20 requests/minute "
                "and 50/day on an unpaid account. Wait a moment and retry."
            )
        return f"Rate limited by {label} (429). Wait a moment and retry."
    if status == 404:
        return f"Model not found on {label} (404). It may have rotated out. Body: {body[:200]}"
    return f"{label} returned HTTP {status}: {body[:300]}"


DEFAULT_SYSTEM_PROMPT = (
    "You are DarkGPT, a precise technical assistant. Answer the question that was "
    "asked, in Markdown, without filler. When information came from an attached file, "
    "ground the answer in that file and say so; never invent file contents. Use the "
    "earlier turns of the conversation - the user expects you to remember them."
)

# Small free models often cap out at 8k-32k tokens. Keep the replayed history
# inside this many characters (~1 token per 4 chars) and drop the oldest turns.
HISTORY_CHAR_BUDGET = 48_000


def _prepare(model: Optional[str], api_key: Optional[str]):
    cfg = provider_config()
    headers = auth_headers(api_key)
    return (model or cfg["default_model"]).strip(), headers


def _user_content(prompt: str, attachments: Optional[List[Dict]]):
    """Plain string unless images are present, then OpenAI content parts."""
    attachments = [a for a in (attachments or []) if not a.get("error")]
    text_blocks = []
    images = []
    for item in attachments:
        if item.get("kind") == "text" and item.get("text"):
            text_blocks.append(f"===== FILE: {item['name']} =====\n{item['text']}")
        elif item.get("kind") == "image" and item.get("data_url"):
            images.append(item)

    message_text = prompt
    if text_blocks:
        message_text = (
            "The user attached the following file(s). Treat them as context for the "
            "question that follows.\n\n"
            + "\n\n".join(text_blocks)
            + f"\n\n===== USER QUESTION =====\n{prompt}"
        )

    if not images:
        return message_text
    content: List[Dict] = [{"type": "text", "text": message_text}]
    for item in images:
        content.append({"type": "image_url", "image_url": {"url": item["data_url"]}})
    return content


def _content_with_attachments(text: str, attachments: Optional[List[Dict]]) -> object:
    """Build message content, including stored text/image attachments."""
    attachments = [a for a in (attachments or []) if not a.get("error")]
    text_blocks = []
    images = []
    for item in attachments:
        if item.get("kind") == "text" and item.get("text"):
            text_blocks.append(f"===== FILE: {item['name']} =====\n{item['text']}")
        elif item.get("kind") == "image" and item.get("data_url"):
            images.append(item)

    message_text = text or ""
    if text_blocks:
        body = "\n\n".join(text_blocks)
        if message_text:
            message_text = (
                "Attached file(s) for context:\n\n"
                + body
                + f"\n\n===== USER MESSAGE =====\n{message_text}"
            )
        else:
            message_text = "Attached file(s) for context:\n\n" + body

    if not images:
        return message_text
    parts: List[Dict] = [{"type": "text", "text": message_text or "See attached image(s)."}]
    for item in images:
        parts.append({"type": "image_url", "image_url": {"url": item["data_url"]}})
    return parts


def _history_content(item: Dict) -> object:
    """Replay a stored turn, reattaching any persisted files/images."""
    text = (item.get("content") or "").strip()
    atts = item.get("attachments") or []
    if not atts:
        return text
    # Only rehydrate if there is actual content to reattach.
    has_text = any(a.get("kind") == "text" and a.get("text") for a in atts)
    has_img = any(a.get("kind") == "image" and a.get("data_url") for a in atts)
    if not has_text and not has_img:
        return text
    return _content_with_attachments(text, atts)


def build_messages(
    history: Optional[List[Dict]] = None,
    prompt: Optional[str] = None,
    attachments: Optional[List[Dict]] = None,
    system: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    max_chars: int = HISTORY_CHAR_BUDGET,
) -> List[Dict]:
    """Stored turns + this prompt, as an OpenAI-style messages list.

    History items may include persisted attachments (text/data_url) so follow-up
    turns can still reference earlier files/images.
    """
    messages: List[Dict] = []
    if system:
        messages.append({"role": "system", "content": system})

    turns: List[Dict] = []
    for item in history or []:
        role = item.get("role")
        if role == "bot":
            role = "assistant"
        if role not in ("user", "assistant"):
            continue
        content = _history_content(item)
        if isinstance(content, str):
            if content:
                turns.append({"role": role, "content": content})
        else:
            turns.append({"role": role, "content": content})

    def _size(turn: Dict) -> int:
        c = turn.get("content")
        if isinstance(c, str):
            return len(c)
        # rough estimate for multipart content
        total = 0
        for part in c if isinstance(c, list) else []:
            if part.get("type") == "text":
                total += len(part.get("text") or "")
            elif part.get("type") == "image_url":
                total += 4000
        return total

    total = sum(_size(t) for t in turns)
    while len(turns) > 1 and total > max_chars:
        total -= _size(turns[0])
        turns.pop(0)
    messages.extend(turns)

    if prompt is not None:
        messages.append({"role": "user", "content": _content_with_attachments(prompt, attachments)})
    return messages


def chat(
    messages: List[Dict],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 180,
) -> str:
    """One blocking completion for a full messages list."""
    model, headers = _prepare(model, api_key)
    label = _provider_label()
    payload = {"model": model, "messages": messages}

    try:
        resp = post_with_retry(chat_url(), headers=headers, payload=payload, timeout=timeout)
    except requests.Timeout:
        raise OpenRouterError(f"{label} did not answer within {timeout}s. Retry or pick another model.")
    except requests.RequestException as exc:
        raise OpenRouterError(f"Could not reach {label} (network/DNS): {exc}")

    if resp.status_code != 200:
        raise parse_error_response(resp)

    try:
        data = resp.json()
    except ValueError:
        raise OpenRouterError(f"{label} returned a non-JSON response.")

    if isinstance(data, dict) and data.get("error"):
        detail = data["error"].get("message") if isinstance(data["error"], dict) else str(data["error"])
        raise OpenRouterError(f"{label} error: {detail}")

    choices = (data or {}).get("choices") or []
    if not choices:
        raise OpenRouterError(f"{label} returned no choices. Raw: {str(data)[:300]}")

    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        reasoning = (message.get("reasoning") or "").strip()
        if reasoning:
            return f"(reasoning only)\n{reasoning}"
        raise OpenRouterError("Model returned an empty message. Try another free model.")
    return content


def chat_stream(
    messages: List[Dict],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 300,
) -> Iterator[str]:
    """Yield the reply token-by-token, for st.write_stream."""
    model, headers = _prepare(model, api_key)
    label = _provider_label()
    payload = {"model": model, "messages": messages, "stream": True}

    try:
        resp = post_with_retry(
            chat_url(),
            headers=headers,
            payload=payload,
            stream=True,
            timeout=(15, timeout),
        )
    except requests.Timeout:
        raise OpenRouterError(f"{label} did not answer within {timeout}s. Retry or pick another model.")
    except requests.RequestException as exc:
        raise OpenRouterError(f"Could not reach {label} (network/DNS): {exc}")

    if resp.status_code != 200:
        try:
            raise parse_error_response(resp)
        finally:
            resp.close()

    emitted = False
    try:
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("error"):
                detail = event["error"].get("message") if isinstance(event["error"], dict) else str(event["error"])
                raise OpenRouterError(f"{label} error: {detail}")
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    emitted = True
                    yield piece
    finally:
        resp.close()

    if not emitted:
        raise OpenRouterError(
            "The model streamed no text. Some free models refuse streaming - retry or pick another."
        )


def get_bot_response(
    prompt: str,
    internal_model: Optional[str] = None,
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 180,
    attachments: Optional[List[Dict]] = None,
    history: Optional[List[Dict]] = None,
) -> str:
    """Single blocking reply. Kept for callers/tests that do not stream."""
    messages = build_messages(history=history, prompt=prompt, attachments=attachments)
    return chat(messages, model=internal_model, api_key=api_key, timeout=timeout)



if __name__ == "__main__":
    models = list_free_models()
    print(f"{len(models)} free text models:")
    for entry in models:
        print("  ", entry["id"], f"({entry['context']} ctx)")
    print("key configured:", bool(get_api_key()))
