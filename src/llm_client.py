"""Generative AI API client: Anthropic (Claude), OpenAI or Google Gemini.

Choose with the LLM_PROVIDER environment variable (default: anthropic). API keys are read from
environment variables / a local .env file, never from code, and are never printed or logged.
Transient failures (rate limits, timeouts, network, 5xx) are retried with exponential backoff.
"""
import os
import re
import time

from dotenv import load_dotenv

from src import config

load_dotenv()


class LLMError(RuntimeError):
    """Friendly, user-displayable error (never contains the API key)."""


def provider() -> str:
    p = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    if p not in config.KEY_VARS:
        raise LLMError(f"Unknown LLM_PROVIDER '{p}'. Choose one of: {', '.join(config.KEY_VARS)}.")
    return p


def get_api_key(prov: str = None):
    prov = prov or provider()
    key = os.getenv(config.KEY_VARS[prov], "").strip()
    return None if (not key or key.lower().startswith("your_")) else key


def llm_status() -> tuple:
    """(is_available, message)"""
    try:
        prov = provider()
    except LLMError as exc:
        return False, str(exc)
    if get_api_key(prov) is None:
        return False, f"No API key found. Set {config.KEY_VARS[prov]} in your .env file (see .env.example)."
    return True, f"Using {prov} model '{model_name(prov)}'."


def model_name(prov: str) -> str:
    return os.getenv("LLM_MODEL", "").strip() or config.DEFAULT_MODELS[prov]


# --------------------------------------------------------------------------- provider adapters
def _call_anthropic(key, model, system, messages, max_tokens):
    import anthropic
    client = anthropic.Anthropic(api_key=key, timeout=config.LLM_TIMEOUT_S, max_retries=0)
    resp = client.messages.create(model=model, max_tokens=max_tokens, system=system, messages=messages)
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def _call_openai(key, model, system, messages, max_tokens):
    from openai import OpenAI
    client = OpenAI(api_key=key, timeout=config.LLM_TIMEOUT_S, max_retries=0)
    resp = client.chat.completions.create(model=model, max_completion_tokens=max_tokens,
                                          messages=[{"role": "system", "content": system}] + messages)
    return resp.choices[0].message.content or ""


def _call_gemini(key, model, system, messages, max_tokens):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key)
    contents = [types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part(text=m["content"])]) for m in messages]
    # Newer Gemini models "think" first, and thinking tokens count against the output limit. Without extra
    # headroom a small limit can be used up entirely by thinking, leaving an empty answer.
    resp = client.models.generate_content(model=model, contents=contents,
                                          config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens + 2048))
    text = getattr(resp, "text", None)
    if not text:
        candidates = getattr(resp, "candidates", None) or []
        reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        block = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
        raise LLMError(f"Gemini returned no text (finish reason: {reason or 'unknown'}{', blocked: ' + str(block) if block else ''}). "
                       "Try again, or set LLM_MODEL to a different Gemini model.")
    return text


ADAPTERS = {"anthropic": _call_anthropic, "openai": _call_openai, "gemini": _call_gemini}
PACKAGES = {"anthropic": "anthropic", "openai": "openai", "gemini": "google-genai"}


def _detail(exc: Exception, key: str = "") -> str:
    """Short error description with any API key removed."""
    text = str(exc).replace("\n", " ")[:220]
    if key:
        text = text.replace(key, "***")
    return re.sub(r"AIza[0-9A-Za-z_-]{10,}|sk-[0-9A-Za-z_-]{10,}", "***", text)


def classify_error(exc: Exception) -> str:
    """'auth' (never retry), 'model' (wrong/retired model name), 'retry' (transient) or 'fatal'."""
    name = type(exc).__name__.lower()
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    msg = str(exc).lower()
    if code in (401, 403) or any(k in name for k in ("authentication", "permissiondenied", "unauthenticated")):
        return "auth"
    if code == 400 and ("api key" in msg or "api_key" in msg):          # Gemini reports a bad key as HTTP 400
        return "auth"
    if code == 404 or "not found" in msg or "no longer available" in msg or "deprecated" in msg:
        return "model"
    if code in (429, 500, 502, 503, 504, 529) or any(k in name for k in ("ratelimit", "timeout", "connection", "unavailable", "overloaded", "internalserver", "servererror")):
        return "retry"
    return "fatal"


def ask(system: str, messages: list, max_tokens: int = 700) -> str:
    """Send a request to the configured provider. Raises LLMError with a friendly message on failure."""
    prov = provider()
    key = get_api_key(prov)
    if key is None:
        raise LLMError(f"No API key found. Set {config.KEY_VARS[prov]} in your .env file (see .env.example).")
    model = model_name(prov)
    for attempt in range(1, config.MAX_LLM_RETRIES + 1):
        try:
            text = ADAPTERS[prov](key, model, system, messages, max_tokens).strip()
            if not text:
                raise LLMError("The AI service returned an empty response.")
            return text
        except LLMError:
            raise
        except ImportError as exc:
            raise LLMError(f"The '{PACKAGES[prov]}' package is not installed. Run: pip install {PACKAGES[prov]}") from exc
        except Exception as exc:                       # provider SDK errors differ, so they are classified by type/status
            kind = classify_error(exc)
            if kind == "auth":
                raise LLMError(f"Authentication failed for {prov}: please check your API key.") from None
            if kind == "model":
                raise LLMError(f"Model '{model}' was not found or has been retired. Set LLM_MODEL in your .env file to a current model name "
                               f"from the {prov} model list. ({_detail(exc, key)})") from None
            if kind == "retry" and attempt < config.MAX_LLM_RETRIES:
                time.sleep(2 ** attempt)               # 2 s, then 4 s
                continue
            if kind == "retry":
                raise LLMError(f"The {prov} service is unavailable or rate-limited (free plans have strict limits). Please wait a minute and try again. ({_detail(exc, key)})") from None
            raise LLMError(f"The {prov} request failed: {type(exc).__name__}: {_detail(exc, key)}") from None
