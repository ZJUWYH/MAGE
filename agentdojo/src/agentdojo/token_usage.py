import math
from collections.abc import Sequence
from typing import Any, Literal

from agentdojo.types import ChatMessage, get_text_content_as_str

TokenScope = Literal["original", "defense"]
TOKEN_USAGE_SCOPE_KEY = "token_usage_scope"


def _new_bucket() -> dict[str, int]:
    return {"input": 0, "output": 0, "total": 0}


def ensure_token_usage(extra_args: dict[str, Any]) -> dict[str, Any]:
    token_usage = extra_args.get("token_usage")
    if not isinstance(token_usage, dict):
        token_usage = {}
    token_usage.setdefault("original", _new_bucket())
    token_usage.setdefault("defense", _new_bucket())
    token_usage.setdefault("total", _new_bucket())

    for scope in ["original", "defense", "total"]:
        bucket = token_usage.get(scope)
        if not isinstance(bucket, dict):
            token_usage[scope] = _new_bucket()
            continue
        bucket.setdefault("input", 0)
        bucket.setdefault("output", 0)
        bucket.setdefault("total", 0)

    extra_args["token_usage"] = token_usage
    return token_usage


def _refresh_total(token_usage: dict[str, Any]) -> None:
    original = token_usage["original"]
    defense = token_usage["defense"]
    total = token_usage["total"]
    total["input"] = int(original["input"]) + int(defense["input"])
    total["output"] = int(original["output"]) + int(defense["output"])
    total["total"] = int(original["total"]) + int(defense["total"])


def get_token_scope(extra_args: dict[str, Any]) -> TokenScope:
    scope = extra_args.get(TOKEN_USAGE_SCOPE_KEY, "original")
    return "defense" if scope == "defense" else "original"


def add_token_usage(
    extra_args: dict[str, Any],
    input_tokens: int = 0,
    output_tokens: int = 0,
    *,
    scope: TokenScope | None = None,
    approximate: bool = False,
) -> None:
    token_usage = ensure_token_usage(extra_args)
    scope = scope or get_token_scope(extra_args)
    bucket = token_usage[scope]

    input_tokens = max(0, int(input_tokens))
    output_tokens = max(0, int(output_tokens))
    total_tokens = input_tokens + output_tokens

    bucket["input"] += input_tokens
    bucket["output"] += output_tokens
    bucket["total"] += total_tokens

    if approximate:
        meta = token_usage.setdefault("meta", {})
        approx_counts = meta.setdefault("approximate_calls", {"original": 0, "defense": 0})
        approx_counts[scope] = int(approx_counts.get(scope, 0)) + 1

    _refresh_total(token_usage)


def set_scope(extra_args: dict[str, Any], scope: TokenScope) -> None:
    extra_args[TOKEN_USAGE_SCOPE_KEY] = scope


def clear_scope(extra_args: dict[str, Any]) -> None:
    extra_args.pop(TOKEN_USAGE_SCOPE_KEY, None)


def estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0

    try:
        import tiktoken  # type: ignore

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        pass

    try:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        pass

    return max(1, math.ceil(len(text) / 4))


def estimate_tokens_from_messages(messages: Sequence[ChatMessage]) -> int:
    pieces: list[str] = []
    for message in messages:
        if message.get("content") is not None:
            pieces.append(get_text_content_as_str(message["content"]))
        tool_calls = message.get("tool_calls")
        if tool_calls:
            for tool_call in tool_calls:
                pieces.append(tool_call.function)
                if tool_call.args:
                    pieces.append(str(dict(tool_call.args)))
    return estimate_tokens_from_text("\n".join(pieces))
