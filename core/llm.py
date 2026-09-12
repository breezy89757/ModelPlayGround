"""任何 OpenAI 相容端點的薄封裝。

刻意不綁任何 provider —— Azure OpenAI、OpenAI、Groq、Together、OpenRouter、
Ollama（/v1）都能跑，只要那個端點吃 /chat/completions。

重點：
- reasoning_effort / response_format 不是每家都支援，收到 400 就把該參數拔掉重試，
  而不是整個失敗。這樣同一份程式碼才能跨 provider
- 每次呼叫都量 TTFT 與總耗時
"""
import os
import json
import time
import asyncio
from dataclasses import dataclass, field

from openai import AsyncOpenAI, BadRequestError

_client: AsyncOpenAI | None = None


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        base = _env("LLM_BASE_URL", "AZURE_OPENAI_ENDPOINT").rstrip("/")
        key = _env("LLM_API_KEY", "AZURE_OPENAI_API_KEY", "OPENAI_API_KEY")
        if not base:
            raise RuntimeError("LLM_BASE_URL 沒有設定，請看 .env.example")
        if not key:
            raise RuntimeError("LLM_API_KEY 沒有設定，請看 .env.example")
        _client = AsyncOpenAI(base_url=base, api_key=key, timeout=300.0)
    return _client


def reset_client() -> None:
    global _client
    _client = None


@dataclass
class Result:
    text: str = ""
    ttft: float = 0.0          # 首字延遲（秒）
    total: float = 0.0         # 總耗時（秒）
    in_tokens: int = 0
    out_tokens: int = 0
    error: str | None = None
    meta: dict = field(default_factory=dict)


async def complete(
    model: str,
    messages: list[dict],
    effort: str | None = None,
    json_mode: bool = False,
    max_tokens: int | None = None,
    on_delta=None,
) -> Result:
    """串流呼叫一次模型。on_delta(text) 會在每個 chunk 被呼叫。"""
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if effort:
        kwargs["reasoning_effort"] = effort
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if max_tokens:
        kwargs["max_completion_tokens"] = max_tokens

    res = Result()
    # 設定錯誤要立刻炸，不要被下面的重試迴圈吞掉當成網路問題
    c = client()
    start = time.perf_counter()

    for attempt in range(3):
        try:
            stream = await c.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.usage:
                    res.in_tokens = chunk.usage.prompt_tokens or 0
                    res.out_tokens = chunk.usage.completion_tokens or 0
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                piece = getattr(delta, "content", None)
                if piece:
                    if res.ttft == 0.0:
                        res.ttft = time.perf_counter() - start
                    res.text += piece
                    if on_delta:
                        on_delta(piece)
            res.total = time.perf_counter() - start
            return res

        except BadRequestError as e:
            # 這個 deployment 不吃某個參數就拔掉重試，而不是整個失敗
            msg = str(e)
            dropped = False
            for param in ("reasoning_effort", "response_format", "stream_options"):
                if param in msg and param in kwargs:
                    kwargs.pop(param)
                    dropped = True
                    break
            if not dropped:
                res.error = msg[:300]
                res.total = time.perf_counter() - start
                return res

        except Exception as e:  # 網路 / 限流
            if attempt == 2:
                res.error = f"{type(e).__name__}: {str(e)[:300]}"
                res.total = time.perf_counter() - start
                return res
            await asyncio.sleep(1.5 * (attempt + 1))

    res.total = time.perf_counter() - start
    return res


async def complete_json(model: str, system: str, user: str, effort: str | None = None) -> dict:
    """要求模型回 JSON，容忍它偶爾包在 ``` 裡面。"""
    r = await complete(
        model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        effort=effort,
        json_mode=True,
    )
    if r.error:
        raise RuntimeError(r.error)
    return _parse_json(r.text)


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s >= 0 and e > s:
            return json.loads(text[s : e + 1])
        raise
