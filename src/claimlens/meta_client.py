"""Meta Model API client: structured calls, caching, usage hooks.

``structured()`` is the single path for schema-constrained model calls: it
hashes the full input, serves ``model_cache`` hits without a network call,
validates every response against its Pydantic model (spec invariant I1),
retries once with a schema-repair follow-up, and raises ``SchemaViolation``
rather than returning a guessed value.

Transports are injectable (``transport=`` stub) so validation, retry, and
cache behavior are unit-testable without an API key. The default transport
targets chat-completions-with-JSON-schema as the most widely supported
OpenAI-compatible mapping; the file-attachment mapping and model/endpoint IDs
are PROVISIONAL until confirmed against the live API at the Issue #1 gate.
``segment()`` additionally needs its live response shape confirmed, so its
default transport raises ``NotImplementedError`` with that pointer.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from . import config, db

log = logging.getLogger(__name__)


class SchemaViolation(Exception):
    """A model response failed schema validation, even after one repair retry."""


class MetaClientError(Exception):
    """Client misconfiguration or API transport failure."""


@dataclass
class FileAttachment:
    """One binary input (photo bytes or estimate PDF)."""

    content: bytes
    mime_type: str  # image/jpeg | image/png | application/pdf
    name: str = ""


@dataclass
class SegmentMask:
    """One SAM part/damage match: label, box, and encoded mask bytes."""

    label: str
    x0: int
    y0: int
    x1: int
    y1: int
    mask: bytes


#: Transport contract: payload dict -> {"data": <parsed JSON>, "usage": {...}?}.
Transport = Callable[[dict[str, Any]], dict[str, Any]]

#: Usage hook contract: {"kind", "model", "cached", "usage"} per call.
UsageHook = Callable[[dict[str, Any]], None]

_client: Any = None


def get_client() -> Any:
    """Build (once) the OpenAI-compatible client for the Meta Model API."""
    global _client
    if _client is None:
        if not config.META_MODEL_BASE_URL or not config.META_MODEL_API_KEY:
            raise MetaClientError(
                "Set META_MODEL_API_KEY and META_MODEL_BASE_URL to call the API."
            )
        from openai import OpenAI

        _client = OpenAI(
            base_url=config.META_MODEL_BASE_URL,
            api_key=config.META_MODEL_API_KEY,
        )
    return _client


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize a JSON schema for strict structured output.

    The API requires every object to set ``additionalProperties: false`` and
    to list all its properties as required. This only shapes what the model
    must emit; validation still runs against the Pydantic model.
    """
    import copy

    node = copy.deepcopy(schema)

    def walk(n: Any) -> None:
        if isinstance(n, dict):
            if n.get("type") == "object" or "properties" in n:
                n["additionalProperties"] = False
                if isinstance(n.get("properties"), dict):
                    n["required"] = sorted(n["properties"])
            for key in ("properties", "$defs", "definitions"):
                sub = n.get(key)
                if isinstance(sub, dict):
                    for v in sub.values():
                        walk(v)
            for key in ("items", "not"):
                sub = n.get(key)
                if isinstance(sub, dict):
                    walk(sub)
                elif isinstance(sub, list):
                    for v in sub:
                        walk(v)
            for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
                sub = n.get(key)
                if isinstance(sub, list):
                    for v in sub:
                        walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(node)
    return node


def _cache_key(
    kind: str,
    model: str,
    effort: str,
    schema: dict[str, Any],
    messages: list[dict[str, Any]],
    files: list[FileAttachment],
) -> str:
    canonical = json.dumps(
        {
            "kind": kind,
            "model": model,
            "effort": effort,
            "schema": schema,
            "messages": messages,
            "files": [
                {
                    "mime": f.mime_type,
                    "sha256": hashlib.sha256(f.content).hexdigest(),
                }
                for f in files
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _attach_files(
    messages: list[dict[str, Any]], files: list[FileAttachment]
) -> list[dict[str, Any]]:
    """Append file parts to the last user message (PROVISIONAL mapping).

    Images ride as data URLs (standard chat-completions shape). The PDF
    input_file part is Responses-API style and must be confirmed live; if the
    endpoint rejects it, this one function is where the mapping changes.
    """
    if not files:
        return messages
    out = [dict(m) for m in messages]
    idx = max(i for i, m in enumerate(out) if m.get("role") == "user")
    content = out[idx].get("content", "")
    parts = (
        [{"type": "text", "text": content}]
        if isinstance(content, str)
        else list(content)
    )
    for f in files:
        blob = base64.b64encode(f.content).decode("ascii")
        if f.mime_type.startswith("image/"):
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{f.mime_type};base64,{blob}"},
                }
            )
        else:
            parts.append(
                {"type": "input_file", "mime_type": f.mime_type, "data": blob}
            )
    out[idx]["content"] = parts
    return out


def _default_transport(payload: dict[str, Any]) -> dict[str, Any]:
    """Call chat-completions with a JSON-schema response format (PROVISIONAL)."""
    client = get_client()
    try:
        response = client.chat.completions.create(
            model=payload["model"],
            messages=_attach_files(
                payload["messages"], payload.get("file_objs", [])
            ),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": payload["kind"],
                    "strict": True,
                    "schema": payload["schema"],
                },
            },
            extra_body={"reasoning_effort": payload["effort"]},
        )
    except Exception as exc:  # surface as our error type, keep the cause
        raise MetaClientError(f"API call failed: {exc}") from exc
    usage = getattr(response, "usage", None)
    return {
        "data": json.loads(response.choices[0].message.content or "null"),
        "usage": usage.model_dump() if usage is not None else None,
    }


def structured(
    *,
    kind: str,
    response_model: type[BaseModel],
    model: str,
    effort: str,
    messages: list[dict[str, Any]],
    files: list[FileAttachment] | None = None,
    conn: Any = None,
    transport: Transport | None = None,
    on_usage: UsageHook | None = None,
) -> BaseModel:
    """Run one schema-constrained call; return the validated model instance.

    Cache hits skip the network (logged at INFO). When ``conn`` is omitted a
    connection to the configured DB is opened and closed here; a caller-passed
    connection is left open. ``transport`` overrides the network call (tests).
    """
    files = files or []
    schema = _strict_schema(response_model.model_json_schema())
    key = _cache_key(kind, model, effort, schema, messages, files)

    own_conn = conn is None
    if own_conn:
        conn = db.connect()
        db.init_schema(conn)
    try:
        cached = db.cache_get(conn, key)
        if cached is not None:
            log.info("structured cache HIT kind=%s model=%s", kind, model)
            obj = response_model.model_validate(cached)
            usage: dict[str, Any] | None = None
            was_cached = True
        else:
            log.info("structured cache MISS kind=%s model=%s", kind, model)
            call = transport or _default_transport
            payload = {
                "kind": kind,
                "model": model,
                "effort": effort,
                "schema": schema,
                "messages": messages,
                "file_objs": files,
            }
            result = call(payload)
            try:
                obj = response_model.model_validate(result["data"])
            except ValidationError as first:
                payload["messages"] = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous output was not valid. "
                            f"Errors: {first}. "
                            "Return ONLY JSON matching the schema."
                        ),
                    }
                ]
                log.info("structured schema-repair retry kind=%s", kind)
                result = call(payload)
                try:
                    obj = response_model.model_validate(result["data"])
                except ValidationError as second:
                    raise SchemaViolation(
                        f"{kind} failed schema validation: {second}"
                    ) from second
            db.cache_put(conn, key, obj.model_dump(mode="json"))
            usage = result.get("usage")
            was_cached = False
    finally:
        if own_conn:
            conn.close()
    if on_usage is not None:
        on_usage(
            {"kind": kind, "model": model, "cached": was_cached, "usage": usage}
        )
    return obj


def segment(
    image: bytes,
    phrases: list[str],
    *,
    model: str | None = None,
    transport: Transport | None = None,
) -> list[SegmentMask]:
    """Run SAM segmentation for ``phrases`` on one image.

    Custom transports normalize to ``{"masks": [{label, x0..y1, mask}]}``. The
    default transport is intentionally unimplemented: the live SAM endpoint and
    response shape must be confirmed first (Issue #1 gate).
    """
    if transport is None:
        raise NotImplementedError(
            "Default SAM transport needs the live Meta Model API endpoint and "
            "response shape confirmed (Issue #1 gate). Pass transport= in tests."
        )
    raw = transport(
        {
            "kind": "segment",
            "model": model or config.SAM_MODEL,
            "image_b64": base64.b64encode(image).decode("ascii"),
            "phrases": list(phrases),
        }
    )
    return [SegmentMask(**m) for m in raw["masks"]]
