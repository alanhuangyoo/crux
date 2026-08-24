"""Model client.

Wraps litellm so the agent loop deals in plain text and a usage record, and so
every provider quirk stays in one place.

Harbor hands the agent a ``provider/model`` string (``deepseek/deepseek-chat``,
``anthropic/claude-fable-5``), which is already litellm's own naming, so the
model name passes straight through.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import litellm

# litellm is chatty on import and re-checks the remote model map on some paths;
# neither is wanted inside a graded trial.
litellm.suppress_debug_info = True
litellm.telemetry = False

logger = logging.getLogger(__name__)

# Transient failures are expected at high concurrency. A trial that dies on a
# 429 counts as reward 0 and cannot be excluded from the submission, so it is
# always worth waiting out a rate limit rather than letting it kill the run.
RETRY_EXCEPTIONS = (
    litellm.RateLimitError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.APIConnectionError,
    litellm.Timeout,
)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class ModelReply:
    content: str
    reasoning_content: str | None = None
    usage: Usage = field(default_factory=Usage)


class ModelClient:
    def __init__(
        self,
        model_name: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        request_timeout_sec: int = 300,
        max_retries: int = 5,
        extra_body: dict | None = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout_sec = request_timeout_sec
        self.max_retries = max_retries
        self.extra_body = extra_body or {}

    async def complete(self, messages: list[dict]) -> ModelReply:
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                kwargs: dict = {
                    "model": self.model_name,
                    "messages": messages,
                    "timeout": self.request_timeout_sec,
                }
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature
                if self.max_tokens is not None:
                    kwargs["max_tokens"] = self.max_tokens
                if self.extra_body:
                    kwargs.update(self.extra_body)

                response = await litellm.acompletion(**kwargs)
                return self._parse(response)

            except RETRY_EXCEPTIONS as exc:
                last_error = exc
                # Exponential backoff, capped — a provider-side incident should
                # not turn into a half-hour sleep inside one trial.
                delay = min(2**attempt, 30)
                logger.warning(
                    "model call failed (%s), retry %d/%d in %ds",
                    type(exc).__name__,
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"model call failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def _parse(self, response) -> ModelReply:
        message = response.choices[0].message
        content = message.get("content") if isinstance(message, dict) else message.content
        reasoning = None
        if isinstance(message, dict):
            reasoning = message.get("reasoning_content")
        else:
            reasoning = getattr(message, "reasoning_content", None)

        return ModelReply(
            content=content or "",
            reasoning_content=reasoning,
            usage=self._parse_usage(response),
        )

    def _parse_usage(self, response) -> Usage:
        usage = Usage()
        raw = getattr(response, "usage", None)
        if raw is None:
            return usage

        usage.prompt_tokens = getattr(raw, "prompt_tokens", 0) or 0
        usage.completion_tokens = getattr(raw, "completion_tokens", 0) or 0

        # Cache-hit accounting differs by provider: DeepSeek reports
        # prompt_cache_hit_tokens at the top level, others nest it under
        # prompt_tokens_details.
        cached = getattr(raw, "prompt_cache_hit_tokens", None)
        if cached is None:
            details = getattr(raw, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", None)
        usage.cached_tokens = cached or 0

        try:
            usage.cost_usd = litellm.completion_cost(completion_response=response) or 0.0
        except Exception:
            # An unmapped model is not a reason to fail a trial; Harbor still
            # records token counts, and cost can be reconciled afterwards.
            usage.cost_usd = 0.0

        return usage
