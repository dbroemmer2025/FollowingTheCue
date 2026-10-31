from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


DEFAULT_OPENAI_BASE_URL = None
DEFAULT_OLLAMA_LOCAL_BASE_URL = "http://http://192.168.1.70:11434"
DEFAULT_OLLAMA_CLOUD_BASE_URL = "https://ollama.com/api"


@dataclass(frozen=True)
class ChatResponse:
    request_ts: str
    response_ts: str
    content: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    raw_response: str
    effective_temperature: float
    finish_reason: str | None = None
    refusal: str | None = None


def normalize_model_name(model: str) -> str:
    model = model.strip()
    aliases = {
        "gpt5": "gpt-5",
    }
    return aliases.get(model.lower(), model)


def supports_custom_temperature(model: str) -> bool:
    normalized = normalize_model_name(model).lower()
    # GPT-5 and GPT-6 currently accept only default temperature behavior.
    if normalized.startswith(("gpt-5", "gpt-6")):
        return False
    return True


def split_reasoning_and_final(content: str) -> tuple[str | None, str, str]:
    content = content.strip()
    s_open = "<reasoning_trace>"
    s_close = "</reasoning_trace>"
    f_open = "<final_answer>"
    f_close = "</final_answer>"
    if all(token in content for token in (s_open, s_close, f_open, f_close)):
        scratchpad = content.split(s_open, 1)[1].split(s_close, 1)[0].strip()
        final = content.split(f_open, 1)[1].split(f_close, 1)[0].strip()
        return (scratchpad or None, final, "xml")

    final_markers = [
        "<final_answer>",
        "**Final Answer:**",
        "**Final answer:**",
        "Final Answer:",
        "Final answer:",
        "Answer:",
    ]
    marker_pos: tuple[int, str] | None = None
    for marker in final_markers:
        idx = content.find(marker)
        if idx != -1 and (marker_pos is None or idx < marker_pos[0]):
            marker_pos = (idx, marker)

    if marker_pos is not None:
        idx, marker = marker_pos
        if marker == "<final_answer>":
            tail = content[idx + len(marker):].strip()
            if f_close in tail:
                final = tail.split(f_close, 1)[0].strip()
            else:
                final = tail
        else:
            final = content[idx + len(marker):].strip()
        reasoning = content[:idx].strip()
        return (reasoning or None, final or content, "marker")

    if s_open in content:
        after_open = content.split(s_open, 1)[1]
        if s_close in after_open:
            reasoning = after_open.split(s_close, 1)[0].strip()
            remainder = after_open.split(s_close, 1)[1].strip()
            return (reasoning or None, remainder or content, "reasoning_tag_only")
        return (after_open.strip() or None, content, "reasoning_tag_only")
    return (None, content, "none")


def resolve_api_key(provider: str, api_key: str | None = None) -> str | None:
    clean = (api_key or "").strip()
    if clean:
        return clean
    env_var = "OPENAI_API_KEY" if provider.lower() == "openai" else "OLLAMA_API_KEY"
    from_env = os.getenv(env_var, "").strip()
    return from_env or None


def resolve_ollama_base_url(base_url: str | None, api_key: str | None = None) -> str:
    clean = (base_url or "").strip()
    if clean:
        return _ensure_ollama_api_prefix(clean)
    if resolve_api_key("ollama", api_key):
        return DEFAULT_OLLAMA_CLOUD_BASE_URL
    return _ensure_ollama_api_prefix(DEFAULT_OLLAMA_LOCAL_BASE_URL)


def create_client(
    provider: str,
    base_url: str | None = None,
    api_key: str | None = None,
    http_timeout: float = 300.0,
) -> Any:
    normalized_provider = provider.lower().strip()
    if normalized_provider == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai package is required. Install with: pip install openai") from exc
        client_kwargs: dict[str, Any] = {"timeout": http_timeout, "max_retries": 0}
        if base_url:
            client_kwargs["base_url"] = base_url
        resolved_key = resolve_api_key("openai", api_key)
        if not resolved_key:
            raise RuntimeError("Set OPENAI_API_KEY in your environment before using provider=openai")
        client_kwargs["api_key"] = resolved_key
        return OpenAI(**client_kwargs)
    if normalized_provider == "ollama":
        return {
            "base_url": resolve_ollama_base_url(base_url, api_key=api_key),
            "timeout": http_timeout,
            "api_key": resolve_api_key("ollama", api_key),
        }
    raise ValueError("provider must be one of: openai, ollama")


def call_chat_completion(
    client: Any,
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
    *,
    json_schema: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
) -> ChatResponse:
    normalized_provider = provider.lower().strip()
    if client is None:
        raise RuntimeError("Model client is not available.")
    if normalized_provider == "ollama":
        if json_schema is not None or reasoning_effort is not None:
            raise ValueError("json_schema/reasoning_effort keyword options are OpenAI-only")
        return _call_ollama_chat_completion(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if normalized_provider == "openai":
        return _call_openai_chat_completion(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
            reasoning_effort=reasoning_effort,
        )
    raise ValueError("provider must be one of: openai, ollama")


def _ensure_ollama_api_prefix(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/api"):
        normalized_path = path
    elif path:
        normalized_path = f"{path}/api"
    else:
        normalized_path = "/api"
    return urllib.parse.urlunparse(parsed._replace(path=normalized_path))


def _call_ollama_chat_completion(
    client: dict[str, Any],
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
) -> ChatResponse:
    base_url = str(client.get("base_url", DEFAULT_OLLAMA_LOCAL_BASE_URL)).rstrip("/")
    timeout = float(client.get("timeout", 300.0))
    api_key = resolve_api_key("ollama", client.get("api_key"))

    payload: dict[str, object] = {"model": model, "messages": messages, "stream": False}
    # Optional native Ollama controls; ordinary callers retain existing defaults.
    if client.get("format") is not None:
        payload["format"] = client["format"]
    if client.get("think") is not None:
        payload["think"] = client["think"]
    options: dict[str, object] = {}
    if client.get("num_ctx") is not None:
        options["num_ctx"] = int(client["num_ctx"])
    effective_temperature = temperature
    if supports_custom_temperature(model):
        options["temperature"] = temperature
    else:
        effective_temperature = 1.0
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    if options:
        payload["options"] = options

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request_ts = utc_now()
    req = urllib.request.Request(
        f"{base_url}/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to connect to Ollama at {base_url}: {exc}") from exc
    response_ts = utc_now()
    parsed = json.loads(raw_body)
    content = str(parsed.get("message", {}).get("content", "")).strip()
    prompt_tokens = parsed.get("prompt_eval_count")
    completion_tokens = parsed.get("eval_count")
    total_tokens = None
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        total_tokens = prompt_tokens + completion_tokens
    return ChatResponse(
        request_ts=request_ts,
        response_ts=response_ts,
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        raw_response=raw_body,
        effective_temperature=effective_temperature,
    )


def _call_openai_chat_completion(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
    json_schema: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
) -> ChatResponse:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "store": False,
    }
    effective_temperature = temperature
    if supports_custom_temperature(model):
        kwargs["temperature"] = temperature
    else:
        effective_temperature = 1.0
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens
    if json_schema is not None:
        kwargs["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "e46_review", "strict": True, "schema": json_schema}}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    request_ts = utc_now()
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        # The caller owns retries, so each billable attempt is accounted for.
        raise RuntimeError(f"OpenAI {type(exc).__name__}: {exc}") from exc
    response_ts = utc_now()
    if not response.choices:
        raise RuntimeError("OpenAI returned no choices")
    choice = response.choices[0]
    content = choice.message.content or ""
    usage = response.usage
    raw_response = response.model_dump_json(indent=0) if hasattr(response, "model_dump_json") else str(response)
    return ChatResponse(
        request_ts=request_ts,
        response_ts=response_ts,
        content=content,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        raw_response=raw_response,
        effective_temperature=effective_temperature,
        finish_reason=choice.finish_reason,
        refusal=getattr(choice.message, "refusal", None),
    )
