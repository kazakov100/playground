import base64
import json
import mimetypes
import os
import time
from io import BytesIO
from typing import Any, Callable, Dict, Optional, Tuple

import requests
from PIL import Image


def post_with_retries(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_s: int = 120,
    retries: int = 3,
) -> Dict[str, Any]:
    last_error = "unknown"
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
            if response.status_code == 200:
                return response.json()
            last_error = f"HTTP {response.status_code}: {response.text[:400]}"
        except Exception as exc:
            last_error = str(exc)

        if attempt < retries:
            time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"OpenRouter request failed after {retries} attempts: {last_error}")


def _compress_image_bytes(
    raw: bytes,
    mime: str,
    max_size_mb: float,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> str:
    if progress_cb:
        progress_cb("Processing... compressing image")

    max_bytes = int(max_size_mb * 1024 * 1024)
    if len(raw) <= max_bytes:
        b64 = base64.b64encode(raw).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    img = Image.open(BytesIO(raw))
    if img.mode not in ("RGB",):
        img = img.convert("RGB")

    # Resize first to keep token/cost lower.
    max_dimension = 1600
    if max(img.size) > max_dimension:
        ratio = max_dimension / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    quality = 80
    for _ in range(5):
        out = BytesIO()
        img.save(out, format="JPEG", quality=quality, optimize=True)
        data = out.getvalue()
        if len(data) <= max_bytes or quality <= 40:
            b64 = base64.b64encode(data).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
        quality -= 10

    # Fallback, should rarely happen.
    out = BytesIO()
    img.save(out, format="JPEG", quality=40, optimize=True)
    b64 = base64.b64encode(out.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def image_to_data_url(
    image_path_or_url: str,
    max_size_mb: float = 3.5,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> str:
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        if progress_cb:
            progress_cb("Processing... downloading image")
        response = requests.get(image_path_or_url, timeout=30)
        response.raise_for_status()
        raw = response.content
        mime = response.headers.get("content-type", "image/jpeg")
    else:
        with open(image_path_or_url, "rb") as f:
            raw = f.read()
        mime, _ = mimetypes.guess_type(image_path_or_url)
        if not mime:
            mime = "image/jpeg"

    return _compress_image_bytes(raw, mime, max_size_mb, progress_cb=progress_cb)


def completion_cost_usd(response: Dict[str, Any], model_id: str) -> float:
    """Estimate USD from an OpenRouter chat/completions JSON `response` (uses `usage`)."""
    return _estimate_cost_usd(response.get("usage") or {}, model_id)


def _estimate_cost_usd(usage: Dict[str, Any], model_id: str) -> float:
    # Approximate OpenRouter pricing ($/1M tokens in/out); unknown models use Claude-like defaults.
    rates = {
        "anthropic/claude-sonnet-4.6": (3.0, 15.0),
        "anthropic/claude-sonnet-4.5": (3.0, 15.0),
        "anthropic/claude-sonnet-4": (3.0, 15.0),
        "openai/gpt-4o-mini": (0.15, 0.60),
        "openai/gpt-4.1-mini": (0.40, 1.60),
    }
    in_rate, out_rate = rates.get(model_id, (3.0, 15.0))
    prompt_tokens = float(usage.get("prompt_tokens", 0))
    completion_tokens = float(usage.get("completion_tokens", 0))
    return (prompt_tokens / 1_000_000.0) * in_rate + (completion_tokens / 1_000_000.0) * out_rate


def classify_image(
    api_key: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    image_path_or_url: str,
    temperature: float = 0.1,
    timeout_s: int = 120,
    retries: int = 3,
    max_size_mb: float = 3.5,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Any], float]:
    if not api_key:
        raise RuntimeError("Missing OpenRouter API key.")

    data_url = image_to_data_url(image_path_or_url, max_size_mb=max_size_mb, progress_cb=progress_cb)
    if progress_cb:
        progress_cb("Processing... sending API request")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8501"),
        "X-Title": "CSV Prompt Evaluator",
    }

    json_contract = (
        "Return ONLY valid JSON with keys: "
        '{"decision":"PASS|FAIL","reason":"short reason","visual_checklist":{"vehicle_posture":"..."}}'
    )

    payload = {
        "model": model_id,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt.strip() + "\n\n" + json_contract},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }

    data = post_with_retries(url, headers, payload, timeout_s=timeout_s, retries=retries)
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"Invalid API response: {data}")

    raw_content = choices[0]["message"]["content"]
    try:
        parsed = json.loads(raw_content)
    except Exception:
        # salvage the first JSON block if extra text leaked in
        s = raw_content.find("{")
        e = raw_content.rfind("}")
        if s == -1 or e == -1 or e <= s:
            raise RuntimeError(f"Model output is not valid JSON: {raw_content[:300]}")
        parsed = json.loads(raw_content[s : e + 1])

    decision = str(parsed.get("decision", "")).upper().strip()
    if decision not in ("PASS", "FAIL"):
        raise RuntimeError(f"Model returned invalid decision: {decision}")

    usage = data.get("usage", {})
    cost = _estimate_cost_usd(usage, model_id)
    parsed["decision"] = decision
    return parsed, cost

