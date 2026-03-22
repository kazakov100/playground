import html
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# `core/` package is at repository root (next to `AI photo automation/`). Prepend repo root
# and this folder so imports work when loaded via root `app.py` or `streamlit run` here.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st

from core.openrouter import classify_image, completion_cost_usd, post_with_retries


DEFAULT_MODEL_ID = "anthropic/claude-sonnet-4.6"
NON_REASONING_MODELS = [
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-sonnet-4",
]


def _optimizer_runtime_vision_blurb(evaluation_model_id: str) -> str:
    """Context for the prompt-optimizer LLM: UI-selected vision model; eval is always non-reasoning."""
    return f"""
## Runtime vision model (what the user prompt is paired with)
- **Model in use:** the one **chosen in the app UI** (sidebar). OpenRouter id: `{evaluation_model_id}`.
- **Default when nothing else is selected:** often `{DEFAULT_MODEL_ID}` — but always optimize for **whichever id is above**.
- **Mode (always):** vision calls are **non-reasoning** — single-pass classification (PASS/FAIL + concise reason), **no** extended chain-of-thought or step-by-step scratchpad, regardless of which vision model is selected.
- **Prompt design:** Favor **explicit** visual criteria, edge cases, and tie-breakers the model can apply in **one** forward pass — not procedures that assume hidden multi-step reasoning.
""".strip()


# Use a cheaper text model for prompt optimization (no vision needed).
OPTIMIZER_MODEL_ID = "openai/gpt-4o-mini"

PROMPT_STATE_FILE = os.path.join(os.path.dirname(__file__), ".prompt_state.json")
# Local-only API key cache (never commit — see .gitignore). Not durable on Streamlit Cloud.
API_KEY_LOCAL_FILE = os.path.join(os.path.dirname(__file__), ".openrouter_api_key.json")

# Hard-coded system prompt (not editable in UI)
SYSTEM_PROMPT = """
You are a micromobility parking enforcement officer.
Analyze the parking photo and decide PASS or FAIL.
Return concise reasoning.
""".strip()


def _inject_brand_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ui-bg-top: #7fc5f7;
            --ui-bg-mid: #69b7ef;
            --ui-bg-bot: #79c2f4;
            --ui-text: #0b2333;
            --ui-muted: #5a7488;
            --ui-border: #bfdcf2;
            --ui-card: rgba(255, 255, 255, 0.76);
            --ui-primary-1: #4ab8ff;
            --ui-primary-2: #2288d8;
        }

        .stApp {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif;
            background: radial-gradient(1200px 600px at 50% -10%, #b8e4ff 0%, transparent 55%),
                        linear-gradient(180deg, var(--ui-bg-top) 0%, var(--ui-bg-mid) 52%, var(--ui-bg-bot) 100%);
            color: var(--ui-text) !important;
        }

        .main .block-container {
            background: var(--ui-card);
            backdrop-filter: blur(14px) saturate(130%);
            -webkit-backdrop-filter: blur(14px) saturate(130%);
            border: 1px solid rgba(255, 255, 255, 0.55);
            border-radius: 20px;
            box-shadow: 0 12px 30px rgba(8, 40, 62, 0.12);
            padding: 1.15rem 1.25rem 1.35rem 1.25rem;
        }

        /* Do not target bare `div` — breaks Base Web select (dark chrome + forced text color = invisible label) */
        h1, h2, h3, h4, h5, h6, p, label, span, .stMarkdown, .stCaption {
            color: var(--ui-text) !important;
        }

        .brand-hero {
            border: 1px solid rgba(255, 255, 255, 0.65);
            background: linear-gradient(135deg, rgba(255,255,255,0.85) 0%, rgba(240,249,255,0.86) 100%);
            border-radius: 18px;
            padding: 18px 20px;
            margin-bottom: 16px;
            box-shadow: 0 8px 22px rgba(10, 62, 96, 0.1);
        }

        .brand-title {
            font-size: 1.45rem;
            font-weight: 700;
            letter-spacing: 0.1px;
            color: #0a2940;
        }

        .model-banner {
            padding: 12px 16px;
            border-radius: 14px;
            border: 1px solid var(--ui-border);
            background: rgba(255, 255, 255, 0.88);
            margin-bottom: 14px;
            box-shadow: 0 4px 14px rgba(10, 62, 96, 0.08);
        }

        /* Streamlit default <code> uses a dark chip — force light UI everywhere */
        .stApp .stMarkdown code {
            background: #eef6fc !important;
            color: #0b2333 !important;
            border: 1px solid rgba(11, 35, 51, 0.12);
            border-radius: 6px;
            padding: 0.12em 0.4em;
            font-size: 0.92em;
        }
        .stApp .stMarkdown pre {
            background: #f6fbff !important;
            color: #0b2333 !important;
            border: 1px solid var(--ui-border) !important;
            border-radius: 10px !important;
        }
        .stApp .stMarkdown pre code {
            background: transparent !important;
            color: #0b2333 !important;
            border: none !important;
            padding: 0 !important;
        }
        /* Evaluation model id — larger chip (must follow generic .stMarkdown code) */
        .stApp .stMarkdown .model-banner code {
            display: inline-block;
            margin-top: 0.35rem;
            padding: 0.4rem 0.65rem !important;
            font-size: 1.05rem !important;
            color: #0b2333 !important;
            word-break: break-all;
            background: #eef6fc !important;
            border: 1px solid var(--ui-border) !important;
            border-radius: 10px;
        }

        .csv-instruction-box {
            padding: 14px 18px;
            border-radius: 14px;
            border: 1px solid var(--ui-border);
            background: #ffffff;
            margin-bottom: 16px;
            box-shadow: 0 4px 14px rgba(10, 62, 96, 0.06);
        }
        .csv-instruction-box h3 {
            margin: 0 0 10px 0;
            font-size: 1.15rem;
            color: #0a2940 !important;
        }
        .csv-instruction-box p {
            margin: 0 0 8px 0;
            color: var(--ui-text) !important;
            line-height: 1.45;
        }
        .csv-instruction-box a {
            color: #2288d8 !important;
            font-weight: 600;
        }

        .stButton > button {
            border-radius: 12px;
            padding: 0.45rem 0.95rem;
            border: 1px solid var(--ui-border);
            background: rgba(255, 255, 255, 0.92);
            color: var(--ui-text);
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(98deg, var(--ui-primary-1) 0%, var(--ui-primary-2) 100%);
            color: #ffffff !important;
            border: none;
            font-weight: 650;
            box-shadow: 0 6px 16px rgba(34, 136, 216, 0.3);
        }

        /* Inputs */
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input {
            background: rgba(255, 255, 255, 0.96) !important;
            color: var(--ui-text) !important;
            border: 1px solid var(--ui-border) !important;
            border-radius: 12px !important;
            padding: 0.55rem 0.75rem !important;
            line-height: 1.4 !important;
            -webkit-text-fill-color: var(--ui-text) !important;
            caret-color: var(--ui-text) !important;
        }

        /* Selectbox — full light control (main + sidebar); Base Web often uses dark fill otherwise */
        .stSelectbox [data-baseweb="select"] {
            background: transparent !important;
        }
        .stSelectbox [data-baseweb="select"] > div {
            background: #ffffff !important;
            color: var(--ui-text) !important;
            border: 1px solid var(--ui-border) !important;
            border-radius: 12px !important;
            min-height: 2.5rem !important;
        }
        .stSelectbox [data-baseweb="select"] div,
        .stSelectbox [data-baseweb="select"] span {
            color: var(--ui-text) !important;
            -webkit-text-fill-color: var(--ui-text) !important;
            opacity: 1 !important;
        }
        /* Dropdown menu panel */
        [data-baseweb="popover"] ul,
        [data-baseweb="menu"] {
            background: #ffffff !important;
            color: var(--ui-text) !important;
            border: 1px solid var(--ui-border) !important;
            border-radius: 12px !important;
        }
        [data-baseweb="popover"] li,
        [data-baseweb="menu"] li {
            color: var(--ui-text) !important;
            background: #ffffff !important;
        }
        [data-baseweb="popover"] li:hover,
        [data-baseweb="menu"] li:hover {
            background: #eef6fc !important;
        }
        .stTextArea textarea::placeholder,
        .stTextInput input::placeholder {
            color: var(--ui-muted) !important;
        }

        /* Box-like components */
        [data-testid="stAlert"],
        [data-testid="stDataFrame"],
        [data-testid="stFileUploader"],
        [data-testid="stExpander"],
        [data-baseweb="tab-panel"] {
            background: rgba(255,255,255,0.9) !important;
            border: 1px solid var(--ui-border) !important;
            border-radius: 14px !important;
            padding: 0.4rem !important;
        }

        [data-testid="stFileUploader"] section {
            background: rgba(255,255,255,0.95) !important;
            border: 1px dashed #9acdea !important;
            border-radius: 12px !important;
        }
        [data-testid="stFileUploader"] section *,
        [data-testid="stFileUploader"] button {
            color: var(--ui-text) !important;
            -webkit-text-fill-color: var(--ui-text) !important;
        }
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] button:hover,
        [data-testid="stFileUploader"] button:focus,
        [data-testid="stFileUploader"] button:active {
            background: #ffffff !important;
            color: var(--ui-text) !important;
            -webkit-text-fill-color: var(--ui-text) !important;
            border: 1px solid var(--ui-border) !important;
            box-shadow: none !important;
        }

        [data-testid="stNumberInput"] button,
        [data-testid="stNumberInput"] [role="button"] {
            background: rgba(255,255,255,0.95) !important;
            color: var(--ui-text) !important;
            border: 1px solid var(--ui-border) !important;
            border-radius: 10px !important;
        }

        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary * {
            background: transparent !important;
            color: var(--ui-text) !important;
            line-height: 1.35 !important;
        }

        [data-baseweb="tab-list"] {
            gap: 6px;
            margin-bottom: 6px;
        }
        [data-baseweb="tab"] {
            background: rgba(255,255,255,0.92) !important;
            border: 1px solid var(--ui-border) !important;
            border-radius: 12px 12px 0 0 !important;
            padding: 0.45rem 0.9rem !important;
        }

        /* Left sidebar readability (avoid `*` — it breaks Base Web select selected value) */
        [data-testid="stSidebar"] {
            background: rgba(255,255,255,0.86) !important;
            border-right: 1px solid var(--ui-border) !important;
        }
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--ui-text) !important;
            -webkit-text-fill-color: var(--ui-text) !important;
        }
        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stNumberInput input,
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
            background: #ffffff !important;
            color: var(--ui-text) !important;
            border: 1px solid var(--ui-border) !important;
        }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div,
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span {
            color: var(--ui-text) !important;
            -webkit-text-fill-color: var(--ui-text) !important;
            opacity: 1 !important;
        }

        /* Streamlit chrome (top/right dark boxes) */
        [data-testid="stToolbar"],
        [data-testid="stHeader"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stDeployButton"],
        [data-testid="stSidebarNav"] {
            background: transparent !important;
            box-shadow: none !important;
            border: none !important;
        }

        [data-testid="stToolbar"] *,
        [data-testid="stStatusWidget"] *,
        [data-testid="stDeployButton"] * {
            color: var(--ui-text) !important;
            -webkit-text-fill-color: var(--ui-text) !important;
        }

        /* Hide Streamlit Cloud "Deploy" button (top right) */
        [data-testid="stDeployButton"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_brand_header() -> None:
    st.markdown(
        """
        <div class="brand-hero">
            <div class="brand-title">🛴 Bird AI Parking Validator 🤖</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metabase_csv_instructions() -> None:
    """Metabase → Sheets workflow for building the evaluation CSV."""
    u1 = "https://metabase.svc.bird.co/question/24173-ai-parking-photo-review-evals-overview-modified"
    u2 = "https://metabase.svc.bird.co/question/25013-ride-end-photo-review-get-market-by-ride-id"
    st.markdown(
        f"""
<div class="csv-instruction-box">
<h3>Build your evaluation CSV (before upload)</h3>
<p><strong>1.</strong> Run <a href="{u1}" target="_blank" rel="noopener noreferrer">Images + ride IDs</a> (Metabase) and note the <strong>ride IDs</strong>.</p>
<p><strong>2.</strong> Open <a href="{u2}" target="_blank" rel="noopener noreferrer">Markets by ride</a> and <strong>paste those ride IDs</strong> into the question’s ride-ID filter / parameter, then run it.</p>
<p><strong>3.</strong> <strong>Merge the two result sets offline</strong> in <strong>Google Sheets</strong> (e.g. join on ride ID), shape columns to match this app (<code>image_url</code>, <code>market_name</code>, ground truth), then <strong>download as CSV</strong> and upload in section 1 below.</p>
</div>
        """.strip(),
        unsafe_allow_html=True,
    )


def _render_evaluation_model_banner(model_id: str) -> None:
    """Prominent box showing the active OpenRouter model (section 2)."""
    safe_model = html.escape(str(model_id))
    st.markdown(
        f'<div class="model-banner">'
        f"<strong>Evaluation model</strong><br/>"
        f"<code>{safe_model}</code>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _load_openrouter_key() -> str:
    """Resolve API key from env, then from `.env` near this file or cwd (Streamlit-safe)."""
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if key:
        return key

    app_dir = os.path.dirname(os.path.abspath(__file__))
    candidates: List[str] = []
    seen: Set[str] = set()
    for root in (
        app_dir,
        os.path.dirname(app_dir),
        os.path.dirname(os.path.dirname(app_dir)),
        os.getcwd(),
    ):
        path = os.path.join(root, ".env")
        if not os.path.isfile(path):
            continue
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        candidates.append(path)

    for env_path in candidates:
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    if k.strip() == "OPENROUTER_API_KEY":
                        value = v.strip().strip('"').strip("'")
                        if value:
                            return value
        except OSError:
            continue

    return ""


def _load_api_key_from_disk() -> str:
    """Persisted key from last session (local dev). Empty if missing."""
    if not os.path.isfile(API_KEY_LOCAL_FILE):
        return ""
    try:
        with open(API_KEY_LOCAL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            v = data.get("openrouter_api_key", "")
            return str(v).strip() if v else ""
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return ""


def _save_api_key_to_disk() -> None:
    """Write key to a local json file (chmod 600 on Unix)."""
    k = st.session_state.get("openrouter_api_key", "")
    if not isinstance(k, str):
        k = ""
    k = k.strip()
    try:
        if not k:
            if os.path.isfile(API_KEY_LOCAL_FILE):
                os.remove(API_KEY_LOCAL_FILE)
            return
        with open(API_KEY_LOCAL_FILE, "w", encoding="utf-8") as f:
            json.dump({"openrouter_api_key": k}, f, ensure_ascii=False, indent=2)
        try:
            os.chmod(API_KEY_LOCAL_FILE, 0o600)
        except OSError:
            pass
    except OSError:
        pass


def _api_key_from_streamlit_secrets() -> str:
    """Streamlit Community Cloud / local secrets.toml."""
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            return str(st.secrets["OPENROUTER_API_KEY"]).strip()
    except Exception:
        pass
    return ""


def _get_effective_api_key() -> str:
    """
    Priority: sidebar input (session, includes value loaded from disk) →
    Streamlit Secrets → env / .env files.
    """
    ui = st.session_state.get("openrouter_api_key", "")
    if isinstance(ui, str) and ui.strip():
        return ui.strip()
    s = _api_key_from_streamlit_secrets()
    if s:
        return s
    return _load_openrouter_key()


def _load_prompt_state_from_disk() -> Dict[str, str]:
    if not os.path.exists(PROMPT_STATE_FILE):
        return {}
    try:
        with open(PROMPT_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _save_prompt_state_to_disk() -> None:
    payload = {
        "prompt_name_1": st.session_state.get("prompt_name_1", "Prompt 1"),
        "prompt_name_2": st.session_state.get("prompt_name_2", "Prompt 2"),
        "prompt_name_3": st.session_state.get("prompt_name_3", "Prompt 3"),
        "prompt_1": st.session_state.get("prompt_1", ""),
        "prompt_2": st.session_state.get("prompt_2", ""),
        "prompt_3": st.session_state.get("prompt_3", ""),
    }
    try:
        with open(PROMPT_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        # Non-fatal; UI should continue even if persistence fails.
        pass


def _apply_optimizer_result_to_prompt_slots(
    suggestion_text: str,
    latest: Dict[str, Any],
    latest_prompt_text: str,
) -> None:
    """
    Put the revised optimizer output in **Prompt 1**; keep the pre-optimization prompt
    (the text that was optimized) in **Prompt 2** so it is not lost.
    If Prompt 1 still held different text than the last run (e.g. edited after run), stash that in **Prompt 3**.
    """
    old_p1_full = st.session_state.get("prompt_1") or ""
    old_p1 = old_p1_full.strip()
    src_name = str(latest.get("prompt_name", "Prompt 1")).strip() or "Prompt 1"
    orig = latest_prompt_text or ""
    orig_stripped = orig.strip()

    st.session_state.prompt_1 = suggestion_text
    st.session_state.prompt_name_1 = (
        f"{src_name} — revised" if "— revised" not in src_name else src_name
    )
    st.session_state.prompt_2 = orig
    st.session_state.prompt_name_2 = (
        f"{src_name} — original" if "— original" not in src_name else f"{src_name} (original)"
    )

    if old_p1 and old_p1 != orig_stripped:
        st.session_state.prompt_3 = old_p1_full
        st.session_state.prompt_name_3 = "Prompt 1 — before optimizer (differs from last run)"

    _save_prompt_state_to_disk()


def _apply_suggestion_to_prompt() -> None:
    """Safe callback: apply optimized suggestion to chosen slot and queue immediate run."""
    suggestion = st.session_state.get("optimized_prompt_suggestion", "").strip()
    if not suggestion:
        st.session_state["apply_suggest_success"] = ""
        return

    target_prompt_slot = st.session_state.get("apply_suggest_target_slot", "Prompt 1")
    slot_idx = {"Prompt 1": 1, "Prompt 2": 2, "Prompt 3": 3}.get(target_prompt_slot, 1)
    source_name = st.session_state.get("optimized_prompt_source", f"Prompt {slot_idx}").strip() or f"Prompt {slot_idx}"
    revised_name = source_name if source_name.lower().endswith(" - revised") else f"{source_name} - revised"

    baseline_raw = st.session_state.get("optimized_prompt_baseline")
    baseline = (baseline_raw or "").strip()
    # Applying to Prompt 1: keep the pre-optimizer baseline in Prompt 2 (same as auto-save behavior).
    if slot_idx == 1 and baseline:
        st.session_state.prompt_2 = baseline_raw or ""
        st.session_state.prompt_name_2 = (
            f"{source_name} — original" if "— original" not in source_name else f"{source_name} (original)"
        )

    st.session_state[f"prompt_{slot_idx}"] = suggestion
    st.session_state[f"prompt_name_{slot_idx}"] = revised_name
    st.session_state["pending_suggested_run"] = {
        "prompt_slot": slot_idx,
        "prompt_name": revised_name,
        "prompt_text": suggestion,
    }
    _save_prompt_state_to_disk()
    st.session_state["apply_suggest_success"] = (
        f"Applied suggestion to {target_prompt_slot} and queued `{revised_name}` for a new run."
    )


def _normalize_expected_result(value: Any) -> str:
    text = str(value).strip().upper()
    if text in ("PASS", "FAIL"):
        return text
    return ""


def _normalize_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize user CSV to required columns:
    - image_url
    - market_name
    - ground_truth
    """
    col_map: Dict[str, str] = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        if key in ("image_url", "url", "image"):
            col_map[col] = "image_url"
        elif key in ("market", "market_name"):
            col_map[col] = "market_name"
        elif key in ("expected_result", "ground_truth", "evaluation_status", "expected"):
            col_map[col] = "ground_truth"

    normalized = df.rename(columns=col_map)
    required = {"image_url", "market_name", "ground_truth"}
    if not required.issubset(set(normalized.columns)):
        missing = required - set(normalized.columns)
        raise ValueError(f"Missing required CSV columns: {', '.join(sorted(missing))}")

    normalized = normalized[["image_url", "market_name", "ground_truth"]].copy()
    normalized["image_url"] = normalized["image_url"].astype(str).str.strip()
    normalized["market_name"] = normalized["market_name"].astype(str).str.strip()
    normalized["ground_truth"] = normalized["ground_truth"].map(_normalize_expected_result)
    normalized = normalized[
        (normalized["image_url"] != "")
        & (normalized["market_name"] != "")
        & (normalized["ground_truth"].isin(["PASS", "FAIL"]))
    ].reset_index(drop=True)
    return normalized


def _filter_by_market(df: pd.DataFrame, market_query: str) -> pd.DataFrame:
    if not market_query.strip():
        return df
    q = market_query.strip().lower()
    return df[df["market_name"].str.lower() == q].reset_index(drop=True)


def _compute_metrics(rows: List[Dict[str, Any]], total_cost: float) -> Dict[str, Any]:
    total = len(rows)
    tp = sum(1 for r in rows if r["ground_truth"] == "PASS" and r["pred"] == "PASS")
    tn = sum(1 for r in rows if r["ground_truth"] == "FAIL" and r["pred"] == "FAIL")
    fp = sum(1 for r in rows if r["ground_truth"] == "FAIL" and r["pred"] == "PASS")
    fn = sum(1 for r in rows if r["ground_truth"] == "PASS" and r["pred"] == "FAIL")
    gt_pass = tp + fn
    gt_fail = tn + fp
    accuracy = (tp + tn) / total * 100.0 if total else 0.0
    fpr = fp / (fp + tn) * 100.0 if (fp + tn) else 0.0
    fnr = fn / (fn + tp) * 100.0 if (fn + tp) else 0.0
    markets = sorted({str(r["market_name"]) for r in rows})
    return {
        "accuracy": round(accuracy, 2),
        "fpr": round(fpr, 2),
        "fnr": round(fnr, 2),
        "total_images": total,
        "markets": ", ".join(markets),
        "cost_usd": round(total_cost, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "gt_pass_count": gt_pass,
        "gt_fail_count": gt_fail,
    }


def _summarize_run_delta(
    prompt_name: str,
    old_metrics: Optional[Dict[str, Any]],
    new_metrics: Dict[str, Any],
) -> str:
    """Before→after line for re-runs. Accuracy can stay flat while FP/FN counts trade off."""
    nm = new_metrics
    if not old_metrics:
        return (
            f"**{prompt_name}** — first run: accuracy **{nm['accuracy']:.2f}%** · "
            f"FPR {nm['fpr']:.2f}% · FNR {nm['fnr']:.2f}% · FP {nm['fp']} · FN {nm['fn']}"
        )
    om = old_metrics
    d_acc = float(nm["accuracy"]) - float(om["accuracy"])
    d_fp = int(nm["fp"]) - int(om["fp"])
    d_fn = int(nm["fn"]) - int(om["fn"])
    return (
        f"**{prompt_name}** — **before → after:** accuracy "
        f"{om['accuracy']:.2f}% → {nm['accuracy']:.2f}% (Δ **{d_acc:+.2f}** pp) · "
        f"FPR {om['fpr']:.2f}% → {nm['fpr']:.2f}% · FNR {om['fnr']:.2f}% → {nm['fnr']:.2f}% · "
        f"FP {om['fp']}→{nm['fp']} (Δ{d_fp:+d}) · FN {om['fn']}→{nm['fn']} (Δ{d_fn:+d})"
    )


def _add_aux_cost_to_run(run_idx: int, amount: float) -> None:
    """Add estimated USD from insights / optimizer OpenRouter text calls to a run."""
    if amount <= 0:
        return
    idx = int(run_idx) - 1
    runs = st.session_state.get("run_results") or []
    if not (0 <= idx < len(runs)):
        return
    r = runs[idx]
    r["aux_cost_usd"] = round(float(r.get("aux_cost_usd", 0.0)) + float(amount), 6)


def _apply_ground_truth_update(image_url: str, market_name: str, new_gt: str) -> None:
    """Update GT across loaded datasets and existing run results, then recompute metrics."""
    if new_gt not in ("PASS", "FAIL"):
        return

    # Update source dataframes in session.
    for df_key in ("csv_df", "filtered_df"):
        df = st.session_state.get(df_key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            mask = (df["image_url"] == image_url) & (df["market_name"] == market_name)
            if mask.any():
                df.loc[mask, "ground_truth"] = new_gt
                st.session_state[df_key] = df

    # Update all stored run results and recompute metrics.
    updated_runs = []
    for run in st.session_state.get("run_results", []):
        rows = run.get("rows", [])
        changed = False
        for row in rows:
            if row.get("image_url") == image_url and row.get("market_name") == market_name:
                row["ground_truth"] = new_gt
                changed = True

        if changed:
            prev_cost = float(run.get("metrics", {}).get("cost_usd", 0.0))
            run["metrics"] = _compute_metrics(rows, total_cost=prev_cost)
        updated_runs.append(run)

    st.session_state.run_results = updated_runs


def _build_error_lines(rows: List[Dict[str, Any]], max_items: int = 20) -> str:
    if not rows:
        return "- None"
    lines: List[str] = []
    for r in rows[:max_items]:
        lines.append(
            f"- market={r.get('market_name','')} | pred={r.get('pred','')} | gt={r.get('ground_truth','')} | reason={r.get('reason','')}"
        )
    return "\n".join(lines)


def _build_insight_error_lines(rows: List[Dict[str, Any]], max_items: int = 28) -> str:
    """Richer lines for improvement analysis (includes visual checklist when present)."""
    if not rows:
        return "- None"
    lines: List[str] = []
    for r in rows[:max_items]:
        vc = r.get("visual_checklist") or {}
        posture = ""
        if isinstance(vc, dict):
            posture = str(vc.get("vehicle_posture", "") or "")[:120]
        reason = str(r.get("reason", ""))[:220]
        lines.append(
            f"- market={r.get('market_name', '')} | gt={r.get('ground_truth', '')} pred={r.get('pred', '')} "
            f"| posture={posture!r} | reason={reason!r}"
        )
    return "\n".join(lines)


def _reason_frequency_markdown_for_insights(rows: List[Dict[str, Any]], max_lines: int = 40) -> str:
    """Exact duplicate `reason` strings merged; used so Main failure bullets can end with (N)."""
    from collections import Counter

    if not rows:
        return "- _(no failing rows in this bucket)_"
    keys: List[str] = []
    for r in rows:
        s = str(r.get("reason", "") or "").strip()
        if not s:
            s = "(empty reason)"
        keys.append(s)
    c = Counter(keys)
    lines: List[str] = []
    for reason, count in c.most_common(max_lines):
        disp = reason.replace("\n", " ").replace("|", "/")
        if len(disp) > 420:
            disp = disp[:417] + "…"
        lines.append(f"- **{count}** image(s) — exact model reason: {disp}")
    return "\n".join(lines)


def _compute_market_error_breakdown(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Count FP/FN per market for rows that are wrong."""
    from collections import defaultdict

    agg = defaultdict(lambda: [0, 0])  # fp, fn per market
    for r in rows:
        if r.get("ground_truth") == r.get("pred"):
            continue
        m = str(r.get("market_name", "")).strip() or "(unknown)"
        if r.get("ground_truth") == "FAIL" and r.get("pred") == "PASS":
            agg[m][0] += 1
        elif r.get("ground_truth") == "PASS" and r.get("pred") == "FAIL":
            agg[m][1] += 1
    if not agg:
        return pd.DataFrame(
            columns=[
                "Market",
                "FP: bad Pass (GT Fail → AI Pass)",
                "FN: bad Fail (GT Pass → AI Fail)",
                "Total wrong",
            ]
        )
    out = []
    for m, (fp, fn) in sorted(agg.items(), key=lambda x: -(x[1][0] + x[1][1])):
        out.append(
            {
                "Market": m,
                "FP: bad Pass (GT Fail → AI Pass)": fp,
                "FN: bad Fail (GT Pass → AI Fail)": fn,
                "Total wrong": fp + fn,
            }
        )
    return pd.DataFrame(out)


def _compute_market_error_breakdown_fp_only(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Per-market counts for GP leakage (FP) only — no FN data."""
    from collections import defaultdict

    agg: Dict[str, int] = defaultdict(int)
    for r in rows:
        if r.get("ground_truth") == "FAIL" and r.get("pred") == "PASS":
            m = str(r.get("market_name", "")).strip() or "(unknown)"
            agg[m] += 1
    if not agg:
        return pd.DataFrame(columns=["Market", "GP leakage: bad Pass (GT Fail → AI Pass)"])
    out = [{"Market": m, "GP leakage: bad Pass (GT Fail → AI Pass)": c} for m, c in sorted(agg.items(), key=lambda x: -x[1])]
    return pd.DataFrame(out)


def _compute_market_error_breakdown_fn_only(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Per-market counts for UX friction (FN) only — no FP data."""
    from collections import defaultdict

    agg: Dict[str, int] = defaultdict(int)
    for r in rows:
        if r.get("ground_truth") == "PASS" and r.get("pred") == "FAIL":
            m = str(r.get("market_name", "")).strip() or "(unknown)"
            agg[m] += 1
    if not agg:
        return pd.DataFrame(columns=["Market", "UX friction: bad Fail (GT Pass → AI Fail)"])
    out = [{"Market": m, "UX friction: bad Fail (GT Pass → AI Fail)": c} for m, c in sorted(agg.items(), key=lambda x: -x[1])]
    return pd.DataFrame(out)


def _insights_cache_signature(run_metrics: Dict[str, Any], focus: str = "all") -> str:
    """Invalidate cached LLM insights when relevant metrics change."""
    if focus == "fp":
        return f"fp_{run_metrics.get('fp')}_{run_metrics.get('fpr')}_{run_metrics.get('total_images')}_failv3"
    if focus == "fn":
        return f"fn_{run_metrics.get('fn')}_{run_metrics.get('fnr')}_{run_metrics.get('total_images')}_failv3"
    return (
        f"all_{run_metrics.get('fp')}_{run_metrics.get('fn')}_"
        f"{run_metrics.get('accuracy')}_{run_metrics.get('fpr')}_{run_metrics.get('fnr')}"
        f"_allv6"
    )


def _suggest_improvement_insights(
    api_key: str,
    optimizer_model_id: str,
    runtime_model_id: str,
    system_prompt: str,
    user_prompt: str,
    model_id: str,
    temperature: float,
    metrics: Dict[str, Any],
    fp_rows: List[Dict[str, Any]],
    fn_rows: List[Dict[str, Any]],
    market_breakdown_md: str,
    timeout_s: int,
    retries: int,
    focus: str = "all",
) -> tuple[str, str, float]:
    """LLM: **Main failure points** (headline — reason **(N)**), then three user-prompt actions. focus: 'all', 'fp', 'fn'. Returns (text, model_id, est_cost_usd)."""
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing.")
    if focus not in ("all", "fp", "fn"):
        focus = "all"

    fp_lines = _build_insight_error_lines(fp_rows, max_items=30)
    fn_lines = _build_insight_error_lines(fn_rows, max_items=30)
    fp_reason_freq = _reason_frequency_markdown_for_insights(fp_rows)
    fn_reason_freq = _reason_frequency_markdown_for_insights(fn_rows)

    fixed_system_rule = """
**Constraint:** The **system prompt is fixed** in this product and cannot be edited. The vision **model** and **temperature** are chosen elsewhere — do **not** recommend changing them. Your three actions must **only** suggest concrete edits to the **user prompt** (additions, removals, clearer rules, wording, or checklist items in the user prompt text).

**Balance:** Pass/Fail wording is a **tradeoff** — stricter Pass rules reduce **GP leakage (FP)** but often raise **UX friction (FN)**; more lenient rules do the opposite. Prefer **incremental, targeted** edits (narrow conditions, tie-breakers, edge cases) over sweeping rules that could swing one metric from excellent to catastrophic. When one metric is already **strong** (e.g. very low FPR), call out that fixes for the other side must **not** erase that progress.
""".strip()

    if focus == "fp":
        prompt = f"""
You analyze **GP leakage only** — false positives where ground truth is FAIL but the model predicted PASS.

{fixed_system_rule}

## Run configuration (context only)
- **Model / temperature (do not suggest changing):** {model_id} @ {temperature}
- **System prompt (fixed — read-only context):** {system_prompt[:800]}
- **User prompt (excerpt — this is what you may recommend edits to):** {user_prompt[:2000]}

## Metrics (GP leakage)
- False positive rate (GP leakage): {metrics.get("fpr", 0)}%  (GT=FAIL predicted PASS)
- False negative rate (UX friction): {metrics.get("fnr", 0)}% (context — do not ignore if it is already low)
- FP count: {metrics.get("fp", 0)} | FN count: {metrics.get("fn", 0)} | Total images: {metrics.get("total_images", 0)}

## GP leakage by market
{market_breakdown_md or "No per-market breakdown."}

## False positive samples (model was too lenient)
{fp_lines}

## Reason counts — GP leakage (exact duplicate model explanations merged)
Use these counts for **`(N)`** in Main failure points. Each line is one **exact** `reason` string from the model; the leading number is how many **FP images** share it.
{fp_reason_freq}

---

Output **in this order** (markdown). Be brief in section 1.

### Main failure points
- **2–5 bullets only.** Each bullet **one line** using this exact pattern: `- **1–3 word headline** —` short reason, then **`(N)`** at the **end** of the line (parentheses, N integer). **N** must be the number of **false positive** images covered by that bullet:
  - If the bullet maps to **one** line in **Reason counts** above, **N** = that line’s image count.
  - If the bullet **groups** several exact reasons from that table, **N** = **sum** of those lines’ counts.
  - Do **not** invent counts; every bullet’s **N** must match the table.
- Headline = **bold** 1–3 words; reason text after the em dash ≈ max ~12 words (theme), not a full paste of the table.

Then **three main actions** — focused on reducing **GP leakage (false positives)** via **user prompt** changes. You may briefly note **FN tradeoffs** where stricter Pass rules could backfire.

**Action 1:** One concrete sentence — what to add, remove, or reword in the **user prompt** (highest impact for stricter Pass criteria).

**Action 2:** One concrete sentence (second priority, **user prompt** only).

**Action 3:** One concrete sentence (third priority, **user prompt** only).

Be specific; do not repeat raw tables. Do not mention editing the system prompt, model, or temperature.
At least one action should acknowledge **tradeoffs** if tightening Pass could materially increase FN.
""".strip()
    elif focus == "fn":
        prompt = f"""
You analyze **UX friction only** — false negatives where ground truth is PASS but the model predicted FAIL.

{fixed_system_rule}

## Run configuration (context only)
- **Model / temperature (do not suggest changing):** {model_id} @ {temperature}
- **System prompt (fixed — read-only context):** {system_prompt[:800]}
- **User prompt (excerpt — this is what you may recommend edits to):** {user_prompt[:2000]}

## Metrics (UX friction)
- False negative rate (UX friction): {metrics.get("fnr", 0)}%  (GT=PASS predicted FAIL)
- False positive rate (GP leakage): {metrics.get("fpr", 0)}% (context — do not ignore if it is already low)
- FN count: {metrics.get("fn", 0)} | FP count: {metrics.get("fp", 0)} | Total images: {metrics.get("total_images", 0)}

## UX friction by market
{market_breakdown_md or "No per-market breakdown."}

## False negative samples (model was too strict)
{fn_lines}

## Reason counts — UX friction (exact duplicate model explanations merged)
Use these counts for **`(N)`** in Main failure points. Each line is one **exact** `reason` string from the model; the leading number is how many **FN images** share it.
{fn_reason_freq}

---

Output **in this order** (markdown). Be brief in section 1.

### Main failure points
- **2–5 bullets only.** Each bullet **one line** using this exact pattern: `- **1–3 word headline** —` short reason, then **`(N)`** at the **end** of the line. **N** must match **Reason counts** above (one line’s count, or **sum** if grouping multiple exact reasons). Do **not** invent counts.
- Headline = **bold** 1–3 words; short theme after the em dash (≈ max ~12 words).

Then **three main actions** — focused on reducing **UX friction (false negatives)** via **user prompt** changes. You may briefly note **FP tradeoffs** where looser Fail / more lenient Pass rules could backfire.

**Action 1:** One concrete sentence — what to add, remove, or reword in the **user prompt** (highest impact for fewer bad Fails).

**Action 2:** One concrete sentence (second priority, **user prompt** only).

**Action 3:** One concrete sentence (third priority, **user prompt** only).

Be specific; do not repeat raw tables. Do not mention editing the system prompt, model, or temperature.
At least one action should acknowledge **tradeoffs** if loosening Fail criteria could materially increase FP.
""".strip()
    else:
        prompt = f"""
You analyze vision-model failures for micromobility parking (PASS/FAIL) evaluation.

{fixed_system_rule}

## Run configuration (context only)
- **Model / temperature (do not suggest changing):** {model_id} @ {temperature}
- **System prompt (fixed — read-only context):** {system_prompt[:800]}
- **User prompt (excerpt — this is what you may recommend edits to):** {user_prompt[:2000]}

## Metrics
- Accuracy: {metrics.get("accuracy", 0)}%
- False positive rate (GP leakage): {metrics.get("fpr", 0)}%  (GT=FAIL predicted PASS)
- False negative rate (UX friction): {metrics.get("fnr", 0)}%  (GT=PASS predicted FAIL)
- FP count: {metrics.get("fp", 0)} | FN count: {metrics.get("fn", 0)} | Total images: {metrics.get("total_images", 0)}

## Errors by market (focus here first)
{market_breakdown_md or "No per-market breakdown."}

## False positive samples (model was too lenient)
{fp_lines}

## False negative samples (model was too strict)
{fn_lines}

## Reason counts — GP leakage (exact duplicate model explanations merged)
{fp_reason_freq}

## Reason counts — UX friction (exact duplicate model explanations merged)
{fn_reason_freq}

---

**Full-run “All mistakes” tab (required structure):** This view must reflect **both** failure modes. **Do not** put all three actions on only FP or only FN — split responsibility across actions as below.

**Action 3 is not a summary of Actions 1+2.** It must propose **one integrated, most balanced user-prompt stance** — the single best compromise given the FP and FN patterns above (e.g. one clear tie-breaker hierarchy, one calibration rule, or one scoped “when in doubt” policy that **jointly** limits GP leakage **and** UX friction). Avoid pasting two half-measures; **choose** the approach that optimizes the tradeoff for this run’s error profile, not merely mentioning both risks in parallel.

Output **in this order** (markdown). Be brief in section 1.

### Main failure points
- **GP leakage (wrong Pass):** **1–3** sub-bullets (nested is fine). Each sub-bullet: `- **1–3 word headline** —` short reason, then **`(N)`** at end. **N** = count from **Reason counts — GP leakage** (one row, or **sum** of rows the bullet summarizes). If FP count is 0, write exactly: `No GP leakage samples in this run.`
- **UX friction (wrong Fail):** **1–3** sub-bullets; same pattern; **`(N)`** from **Reason counts — UX friction** table. If FN count is 0, write exactly: `No UX friction samples in this run.`

Then use **this exact markdown structure** for actions (labels must match). Each action must be a **user prompt** edit only.

**Action 1 (GP leakage / false positives):** One concrete sentence — user-prompt edits that reduce **FP** (wrong Pass) using the **FP samples** above. If FP count is 0, write: `No false positives in this run — no GP-specific user-prompt change needed; keep existing FP guardrails.`

**Action 2 (UX friction / false negatives):** One concrete sentence — user-prompt edits that reduce **FN** (wrong Fail) using the **FN samples** above. If FN count is 0, write: `No false negatives in this run — no UX-specific user-prompt change needed; keep existing FN guardrails.`

**Action 3 (Most balanced approach):** One concrete sentence — **synthesize** the best **single** balanced policy from the FP and FN evidence (and metrics): pick the intervention that **best jointly** addresses both failure modes for this run — e.g. scoped rules, ordered tie-breakers, or image-quality gates that **simultaneously** reduce wrong Passes where the model was lenient **and** wrong Fails where it was strict, without contradicting Action 1 or 2. Do **not** restate Action 1 and Action 2 in one line; **select** the most defensible compromise.

If only one error type exists (only FP or only FN), still use all three labels: **Action 1** or **2** gets the “no errors in this run” line for the missing side, and **Action 3** proposes the single best balanced stance for the errors that **do** exist (still not a mere concatenation).

Be specific; do not repeat raw tables. Do not mention editing the system prompt, model, or temperature.
""".strip()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8501"),
        "X-Title": "Bird Improvement Insights",
    }
    model_candidates: List[str] = []
    for m in [optimizer_model_id, "openai/gpt-4o-mini", "openai/gpt-4.1-mini", runtime_model_id]:
        if m and m not in model_candidates:
            model_candidates.append(m)

    last_err = "unknown"
    total_cost = 0.0
    for candidate_model in model_candidates:
        payload = {
            "model": candidate_model,
            "temperature": 0.2,
            "provider": {"allow_fallbacks": True},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a senior ML product engineer who diagnoses vision-classifier failures. "
                        "The system prompt is fixed — recommend only concrete edits to the **user prompt** "
                        "(wording, rules, checklist). Do not recommend changing the system prompt, model, or temperature. "
                        "Always consider **FPR vs FNR tradeoffs**; avoid advice that would obviously collapse one metric "
                        "while fixing the other."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        try:
            data = post_with_retries(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                payload=payload,
                timeout_s=timeout_s,
                retries=retries,
            )
            total_cost += completion_cost_usd(data, candidate_model)
            choices = data.get("choices", [])
            if not choices:
                last_err = f"Empty choices for {candidate_model}"
                continue
            text = (choices[0]["message"]["content"] or "").strip()
            if not text:
                last_err = f"Empty insight body for {candidate_model}"
                continue
            return text, candidate_model, total_cost
        except Exception as exc:
            last_err = f"{candidate_model}: {exc}"
            continue

    raise RuntimeError(f"Improvement insights failed across candidate models. Last error: {last_err}")


def _render_insights_panel(
    *,
    focus: str,
    run_idx: int,
    run_metrics: Dict[str, Any],
    run_result: Dict[str, Any],
    run_model: str,
    market_breakdown_md: str,
    fp_rows: List[Dict[str, Any]],
    fn_rows: List[Dict[str, Any]],
    api_key: str,
    model_id: str,
    temperature: float,
    timeout_s: int,
    retries: int,
    should_run: bool,
) -> None:
    """Generate and show cached LLM insights for one focus: 'fp', 'fn', or 'all'."""
    if not should_run:
        return
    sig = _insights_cache_signature(run_metrics, focus=focus)
    sig_key = f"insights_sig_{run_idx}_{focus}"
    md_key = f"insights_md_{run_idx}_{focus}"
    model_key = f"insights_model_{run_idx}_{focus}"
    need_insights = st.session_state.get(sig_key) != sig

    if not api_key:
        st.info("Set **OPENROUTER_API_KEY** (secrets or `.env`) to show automated insights.")
        return

    if need_insights:
        with st.spinner("Generating insights..."):
            try:
                insight_text, insight_model, insight_cost = _suggest_improvement_insights(
                    api_key=api_key,
                    optimizer_model_id=OPTIMIZER_MODEL_ID,
                    runtime_model_id=model_id,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=run_result.get("prompt_text", ""),
                    model_id=run_model,
                    temperature=float(temperature),
                    metrics=run_metrics,
                    fp_rows=fp_rows,
                    fn_rows=fn_rows,
                    market_breakdown_md=market_breakdown_md,
                    timeout_s=int(timeout_s),
                    retries=int(retries),
                    focus=focus,
                )
                st.session_state[md_key] = insight_text
                st.session_state[model_key] = insight_model
                st.session_state[sig_key] = sig
                _add_aux_cost_to_run(run_idx, insight_cost)
            except Exception as exc:
                st.error(f"Insights unavailable: {exc}")
                return

    cached = st.session_state.get(md_key)
    if cached and st.session_state.get(sig_key) == sig:
        st.caption(f"Model: `{st.session_state.get(model_key, '')}`")
        st.markdown(cached)


def _ensure_insight_cached(
    focus: str,
    run_idx: int,
    run_metrics: Dict[str, Any],
    run_result: Dict[str, Any],
    run_model: str,
    market_breakdown_md: str,
    fp_rows: List[Dict[str, Any]],
    fn_rows: List[Dict[str, Any]],
    api_key: str,
    model_id: str,
    temperature: float,
    timeout_s: int,
    retries: int,
) -> None:
    """Populate session insight cache without UI (used before prompt optimization)."""
    should = (
        (focus == "fp" and fp_rows)
        or (focus == "fn" and fn_rows)
        or (focus == "all" and (fp_rows or fn_rows))
    )
    if not should or not api_key:
        return
    sig = _insights_cache_signature(run_metrics, focus=focus)
    sig_key = f"insights_sig_{run_idx}_{focus}"
    md_key = f"insights_md_{run_idx}_{focus}"
    model_key = f"insights_model_{run_idx}_{focus}"
    if st.session_state.get(sig_key) == sig and st.session_state.get(md_key):
        return
    insight_text, insight_model, insight_cost = _suggest_improvement_insights(
        api_key=api_key,
        optimizer_model_id=OPTIMIZER_MODEL_ID,
        runtime_model_id=model_id,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=run_result.get("prompt_text", ""),
        model_id=run_model,
        temperature=float(temperature),
        metrics=run_metrics,
        fp_rows=fp_rows,
        fn_rows=fn_rows,
        market_breakdown_md=market_breakdown_md,
        timeout_s=int(timeout_s),
        retries=int(retries),
        focus=focus,
    )
    st.session_state[md_key] = insight_text
    st.session_state[model_key] = insight_model
    st.session_state[sig_key] = sig
    _add_aux_cost_to_run(run_idx, insight_cost)


def _ensure_insights_for_optimizer(
    run_idx: int,
    run_metrics: Dict[str, Any],
    run_result: Dict[str, Any],
    run_model: str,
    fp_rows: List[Dict[str, Any]],
    fn_rows: List[Dict[str, Any]],
    err_by_mkt: pd.DataFrame,
    err_fp_mkt: pd.DataFrame,
    err_fn_mkt: pd.DataFrame,
    api_key: str,
    model_id: str,
    temperature: float,
    timeout_s: int,
    retries: int,
) -> None:
    """Ensure fp / fn / all insight markdown exists in session (generates if missing)."""
    mkt_all = err_by_mkt.to_string(index=False) if not err_by_mkt.empty else ""
    mkt_fp = err_fp_mkt.to_string(index=False) if not err_fp_mkt.empty else ""
    mkt_fn = err_fn_mkt.to_string(index=False) if not err_fn_mkt.empty else ""
    if fp_rows or fn_rows:
        _ensure_insight_cached(
            "all",
            run_idx,
            run_metrics,
            run_result,
            run_model,
            mkt_all,
            fp_rows,
            fn_rows,
            api_key,
            model_id,
            temperature,
            timeout_s,
            retries,
        )
    if fp_rows:
        _ensure_insight_cached(
            "fp",
            run_idx,
            run_metrics,
            run_result,
            run_model,
            mkt_fp,
            fp_rows,
            fn_rows,
            api_key,
            model_id,
            temperature,
            timeout_s,
            retries,
        )
    if fn_rows:
        _ensure_insight_cached(
            "fn",
            run_idx,
            run_metrics,
            run_result,
            run_model,
            mkt_fn,
            fp_rows,
            fn_rows,
            api_key,
            model_id,
            temperature,
            timeout_s,
            retries,
        )


def _format_insights_for_optimizer_prompt(objective: str, run_idx: int) -> str:
    """Bundle cached insights for the prompt-optimizer LLM."""
    fp = (st.session_state.get(f"insights_md_{run_idx}_fp") or "").strip()
    fn = (st.session_state.get(f"insights_md_{run_idx}_fn") or "").strip()
    all_ = (st.session_state.get(f"insights_md_{run_idx}_all") or "").strip()
    if not (fp or fn or all_):
        return ""
    parts: List[str] = []
    if all_:
        parts.append("### Full run (FP + FN)\n" + all_)
    if fp:
        parts.append("### GP leakage (false positives)\n" + fp)
    if fn:
        parts.append("### UX friction (false negatives)\n" + fn)
    body = "\n\n".join(parts)
    scope = (
        "**Scope:** The **system prompt is fixed**. Implement these insights only as edits to the **user prompt**. "
        "Where insights conflict, **merge** them with explicit Pass/Fail safeguards so neither FP nor FN explodes.\n\n"
    )
    if objective.startswith("Minimize False Positives"):
        priority = (
            "**Priority:** Implement the **GP leakage** section first; use full-run and UX sections "
            "only where they do not conflict with reducing false Passes."
        )
    elif objective.startswith("Minimize False Negatives"):
        priority = (
            "**Priority:** Implement the **UX friction** section first; use full-run and GP sections "
            "only where they do not conflict with reducing false Fails."
        )
    else:
        priority = (
            "**Priority:** Implement **all** sections below — balance GP and UX guidance in the revised **user** prompt."
        )
    return f"{scope}{priority}\n\n{body}"


def _suggest_prompt_from_errors(
    api_key: str,
    optimizer_model_id: str,
    runtime_model_id: str,
    current_prompt: str,
    fp_rows: List[Dict[str, Any]],
    fn_rows: List[Dict[str, Any]],
    objective: str,
    user_goal_note: str,
    timeout_s: int,
    retries: int,
    insights_markdown: str = "",
    run_metrics: Optional[Dict[str, Any]] = None,
) -> tuple[str, str, float]:
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing.")

    objective_instructions = {
        "Minimize False Positives (GP Leakage)": (
            "Prioritize reducing false positives (GT FAIL predicted PASS), but **do not** do so by "
            "rules that would plausibly spike false negatives (e.g. vague 'when in doubt Fail' if FNR was already acceptable)."
        ),
        "Minimize False Negatives (UX Friction)": (
            "Prioritize reducing false negatives (GT PASS predicted FAIL), but **do not** do so by "
            "rules that would plausibly spike false positives (e.g. 'when in doubt Pass' if FPR was already acceptable)."
        ),
        "Balanced": (
            "Improve both sides together: **reduce** FP and FN without trading one near-zero metric for a catastrophic failure on the other."
        ),
    }
    objective_text = objective_instructions.get(objective, objective_instructions["Balanced"])

    fp_lines = _build_error_lines(fp_rows)
    fn_lines = _build_error_lines(fn_rows)
    max_len = max(300, int(len(current_prompt) * 1.2))
    runtime_vision_block = _optimizer_runtime_vision_blurb(runtime_model_id)

    balance_block = ""
    if run_metrics and isinstance(run_metrics, dict):
        fpr = run_metrics.get("fpr", 0)
        fnr = run_metrics.get("fnr", 0)
        nfp = run_metrics.get("fp", 0)
        nfn = run_metrics.get("fn", 0)
        balance_block = f"""
## Metric snapshot (do not regress the healthy side)
- **FPR (GP leakage):** {fpr}% | **FNR (UX friction):** {fnr}%
- **FP count:** {nfp} | **FN count:** {nfn}

**Balance rules:**
1) Prefer **incremental** clarifications (extra conditions, tie-breakers, edge cases) over a full rewrite that throws away working behavior.
2) If one rate is **already strong** (e.g. FPR near 0%), your revision must **preserve** that — do not introduce broad Pass/Fail shortcuts that would obviously explode the other error type.
3) When diagnostic insights push in one direction, **merge** them with explicit safeguards so the other failure mode does not “drop the ball” entirely.
4) If objective is one-sided, still **hedge**: e.g. “Pass only when …; otherwise Fail” instead of “always Pass when unsure.”

"""

    insights_block = ""
    if insights_markdown.strip():
        insights_block = f"""
## Diagnostic insights (must implement in the revised **user** prompt)
The **system prompt is fixed** and cannot be edited. Translate the following into concrete rules, wording, thresholds, or checklist items in the **user prompt** only. Do not ignore them; resolve conflicts in favor of the stated priority.

{insights_markdown.strip()}

"""
    else:
        insights_block = """
## Diagnostic insights
No separate insight summary was provided — rely on the error samples below. The **system prompt is fixed**; only revise the **user prompt**.

"""

    prompt = f"""
You are an expert prompt engineer for image classification policy prompts.
Generate ONE revised **user prompt** only. The **system prompt is fixed** in the product and must not appear in your output or be modified.

{runtime_vision_block}

Objective:
{objective_text}

Additional user guidance:
{user_goal_note or "None"}

{balance_block}
{insights_block}
Current **user** prompt:
{current_prompt}

Observed False Positives (GT FAIL but predicted PASS):
{fp_lines}

Observed False Negatives (GT PASS but predicted FAIL):
{fn_lines}

Guardrails:
1) Output **only** the revised **user** prompt text — never a system prompt or commentary about the system prompt.
2) Keep it concise and production-ready; preserve original intent; fix error-prone ambiguity.
3) **Implement the diagnostic insights above** when they are present (as **user** prompt edits), but **reconcile** them with the balance rules so the prompt does not swing one metric from excellent to catastrophic.
4) Do not add long examples, JSON schemas, or extra meta-instructions.
5) Keep output length <= {max_len} characters.
6) Output ONLY the revised user prompt text, no commentary.
""".strip()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Prompt Optimizer Assistant",
    }
    model_candidates = []
    # Try cheap text models first, then fall back to the currently selected runtime model.
    for m in [optimizer_model_id, "openai/gpt-4o-mini", "openai/gpt-4.1-mini", runtime_model_id]:
        if m and m not in model_candidates:
            model_candidates.append(m)

    last_err = "unknown optimizer error"
    total_cost = 0.0
    for candidate_model in model_candidates:
        payload = {
            "model": candidate_model,
            "temperature": 0.1,
            # Force compatible providers to avoid provider-policy mismatch.
            # Keep provider preferences broad; account/provider policy may override this.
            "provider": {"allow_fallbacks": True},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You improve the **user** prompt only. The system prompt is fixed — do not output or modify it. "
                        "Preserve product intent. **Balance** FPR and FNR: avoid revisions that fix one failure mode "
                        "while obviously destroying the other (e.g. FPR 0% → 80%). Prefer incremental, testable criteria. "
                        "Vision evaluation uses **whatever model the user selects in the app** (OpenRouter id); "
                        "calls are always **non-reasoning** (standard completion, no extended chain-of-thought). "
                        "User prompts must be **direct and visually grounded**, not dependent on long hidden reasoning."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        try:
            data = post_with_retries(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                payload=payload,
                timeout_s=timeout_s,
                retries=retries,
            )
            total_cost += completion_cost_usd(data, candidate_model)
            choices = data.get("choices", [])
            if not choices:
                last_err = f"Invalid optimizer response for model {candidate_model}: {data}"
                continue
            suggestion = (choices[0]["message"]["content"] or "").strip()
            if not suggestion:
                last_err = f"Empty optimizer output for model {candidate_model}"
                continue
            return suggestion, candidate_model, total_cost
        except Exception as exc:
            last_err = f"{candidate_model}: {exc}"
            continue

    raise RuntimeError(f"Optimizer failed across candidate models. Last error: {last_err}")


def _run_single_prompt(
    prompt_name: str,
    prompt_slot: int,
    user_prompt: str,
    data: pd.DataFrame,
    api_key: str,
    model_id: str,
    temperature: float,
    max_size_mb: float,
    timeout_s: int,
    retries: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    total_cost = 0.0
    progress = st.progress(0.0)
    status = st.empty()
    max_workers = min(10, max(1, len(data)))

    def evaluate_one(row: Any) -> Dict[str, Any]:
        image_url = row.image_url
        market_name = row.market_name
        gt = row.ground_truth
        try:
            out, cost = classify_image(
                api_key=api_key,
                model_id=model_id,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                image_path_or_url=image_url,
                temperature=temperature,
                timeout_s=timeout_s,
                retries=retries,
                max_size_mb=max_size_mb,
                progress_cb=None,
            )
            pred = out.get("decision", "FAIL")
            reason = out.get("reason", "")
            visual_checklist = out.get("visual_checklist") or {}
            if not isinstance(visual_checklist, dict):
                visual_checklist = {}
        except Exception as exc:
            cost = 0.0
            pred = "FAIL"
            reason = f"ERROR: {str(exc)[:180]}"
            visual_checklist = {}

        return {
            "image_url": image_url,
            "market_name": market_name,
            "ground_truth": gt,
            "pred": pred,
            "reason": reason,
            "visual_checklist": visual_checklist,
            "cost": cost,
        }

    total = len(data)
    completed = 0
    status.write(f"Processing... {prompt_name} | starting with {max_workers} workers")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_one, row) for row in data.itertuples(index=False)]
        for fut in as_completed(futures):
            out = fut.result()
            total_cost += float(out.pop("cost", 0.0))
            rows.append(out)
            completed += 1
            progress.progress(completed / total if total else 1.0)
            status.write(f"Processing... {prompt_name} | {completed}/{total}")

    metrics = _compute_metrics(rows, total_cost=total_cost)
    status.success(f"Completed: {prompt_name}")
    return {
        "prompt_name": prompt_name,
        "prompt_slot": prompt_slot,
        "prompt_text": user_prompt,
        "model_id": model_id,
        "rows": rows,
        "metrics": metrics,
        "aux_cost_usd": 0.0,
    }


def _show_run_metric_cards(
    metrics: Dict[str, Any], *, focus: str = "all", aux_cost_usd: float = 0.0
) -> None:
    """focus: 'all' full run metrics; 'fp' GP leakage only (no FN); 'fn' UX friction only (no FP)."""
    vision = float(metrics.get("cost_usd", 0.0))
    aux = float(aux_cost_usd)
    total = vision + aux
    cost_label = "Cost (est.)" if aux > 0 else "Cost"
    cost_val = f"${total:.4f}"
    if focus == "fp":
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2f}%")
        c2.metric("GP leakage rate (FPR)", f"{metrics.get('fpr', 0):.2f}%")
        c3.metric("# False positives (bad Pass)", str(metrics.get("fp", 0)))
        c4.metric(cost_label, cost_val)
        c5.metric("Photos tested", str(metrics.get("total_images", 0)))
        if aux > 0:
            st.caption(
                f"Cost breakdown: vision eval **${vision:.4f}** · insights & optimizer (text, est.) **${aux:.4f}**"
            )
        return
    if focus == "fn":
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2f}%")
        c2.metric("UX friction rate (FNR)", f"{metrics.get('fnr', 0):.2f}%")
        c3.metric("# False negatives (bad Fail)", str(metrics.get("fn", 0)))
        c4.metric(cost_label, cost_val)
        c5.metric("Photos tested", str(metrics.get("total_images", 0)))
        if aux > 0:
            st.caption(
                f"Cost breakdown: vision eval **${vision:.4f}** · insights & optimizer (text, est.) **${aux:.4f}**"
            )
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2f}%")
    c2.metric("False Positive Rate", f"{metrics.get('fpr', 0):.2f}%")
    c3.metric("False Negative Rate", f"{metrics.get('fnr', 0):.2f}%")
    c4.metric(cost_label, cost_val)
    c5.metric("Photos Tested", str(metrics.get("total_images", 0)))
    if aux > 0:
        st.caption(
            f"Cost breakdown: vision eval **${vision:.4f}** · insights & optimizer (text, est.) **${aux:.4f}**"
        )


def _failed_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if r["ground_truth"] != r["pred"]]


def _show_failed_photos_table(rows: List[Dict[str, Any]]) -> bool:
    """Table of all misclassified photos. Returns True if there is at least one failure."""
    failed = _failed_rows(rows)
    if not failed:
        st.info("No failed images for this run.")
        return False
    table_rows = [
        {
            "Photo URL": r["image_url"],
            "Market": r["market_name"],
            "Model prediction": r["pred"],
            "Ground truth (correct label)": r["ground_truth"],
            "Model explanation": (r.get("reason", "") or "")[:2000],
        }
        for r in failed
    ]
    st.markdown("#### Failed photos")
    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        height=min(420, 120 + 28 * len(failed)),
        hide_index=True,
    )
    return True


def _show_failed_gallery(rows: List[Dict[str, Any]], run_key: str) -> None:
    failed = _failed_rows(rows)
    if not failed:
        return
    st.markdown("#### Images")
    st.caption("Use **Set GT** / **Update GT** below each image to correct labels; metrics refresh after update.")
    cols = st.columns(4)
    for idx, item in enumerate(failed):
        with cols[idx % 4]:
            st.markdown(
                (
                    f"**Model prediction:** `{item['pred']}`  \n"
                    f"**Ground truth:** `{item['ground_truth']}`  \n"
                    f"**Market:** `{item['market_name']}`"
                )
            )
            st.image(item["image_url"], width=220)
            st.caption(item.get("reason", ""))
            gt_default = item.get("ground_truth", "FAIL")
            gt_choice = st.selectbox(
                "Set GT",
                options=["PASS", "FAIL"],
                index=0 if gt_default == "PASS" else 1,
                key=f"gt_choice_{run_key}_failed_gallery_{idx}",
            )
            if st.button("Update GT", key=f"gt_btn_{run_key}_failed_gallery_{idx}"):
                _apply_ground_truth_update(
                    image_url=item["image_url"],
                    market_name=item["market_name"],
                    new_gt=gt_choice,
                )
                st.success(
                    f"Updated GT to {gt_choice} for {item['market_name']} | {item['image_url'][:60]}..."
                )
                st.rerun()


def _show_error_bucket(rows: List[Dict[str, Any]], title: str, run_key: str) -> None:
    st.markdown(f"#### {title} ({len(rows)})")
    if not rows:
        st.info(f"No {title.lower()} in this run.")
        return

    table_rows = [
        {
            "Photo URL": r["image_url"],
            "Market": r["market_name"],
            "Model prediction": r["pred"],
            "Ground truth (correct label)": r["ground_truth"],
            "Model explanation": (r.get("reason", "") or "")[:2000],
        }
        for r in rows
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, height=220)

    cols = st.columns(4)
    for idx, item in enumerate(rows):
        with cols[idx % 4]:
            st.markdown(
                (
                    f"**Model prediction:** `{item['pred']}`  \n"
                    f"**Ground truth:** `{item['ground_truth']}`  \n"
                    f"**Market:** `{item['market_name']}`"
                )
            )
            st.image(item["image_url"], width=220)
            st.caption(item.get("reason", ""))
            gt_default = item.get("ground_truth", "FAIL")
            gt_choice = st.selectbox(
                "Set GT",
                options=["PASS", "FAIL"],
                index=0 if gt_default == "PASS" else 1,
                key=f"gt_choice_{run_key}_{title}_{idx}",
            )
            if st.button("Update GT", key=f"gt_btn_{run_key}_{title}_{idx}"):
                _apply_ground_truth_update(
                    image_url=item["image_url"],
                    market_name=item["market_name"],
                    new_gt=gt_choice,
                )
                st.success(
                    f"Updated GT to {gt_choice} for {item['market_name']} | {item['image_url'][:60]}..."
                )
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="CSV Prompt Evaluator", page_icon="📊", layout="wide")
    _inject_brand_css()
    _render_brand_header()
    _render_metabase_csv_instructions()

    if "csv_df" not in st.session_state:
        st.session_state.csv_df = None
    if "filtered_df" not in st.session_state:
        st.session_state.filtered_df = None
    if "run_results" not in st.session_state:
        st.session_state.run_results = []
    if "optimized_prompt_suggestion" not in st.session_state:
        st.session_state.optimized_prompt_suggestion = ""
    if "optimized_prompt_source" not in st.session_state:
        st.session_state.optimized_prompt_source = ""
    if "optimized_prompt_model_used" not in st.session_state:
        st.session_state.optimized_prompt_model_used = ""
    if "optimized_prompt_baseline" not in st.session_state:
        st.session_state.optimized_prompt_baseline = ""
    if "apply_suggest_success" not in st.session_state:
        st.session_state.apply_suggest_success = ""
    if "pending_suggested_run" not in st.session_state:
        st.session_state.pending_suggested_run = None
    persisted = _load_prompt_state_from_disk()
    if "prompt_1" not in st.session_state:
        st.session_state.prompt_1 = persisted.get("prompt_1", "")
    if "prompt_2" not in st.session_state:
        st.session_state.prompt_2 = persisted.get("prompt_2", "")
    if "prompt_3" not in st.session_state:
        st.session_state.prompt_3 = persisted.get("prompt_3", "")
    if "prompt_name_1" not in st.session_state:
        st.session_state.prompt_name_1 = persisted.get("prompt_name_1", "Prompt 1")
    if "prompt_name_2" not in st.session_state:
        st.session_state.prompt_name_2 = persisted.get("prompt_name_2", "Prompt 2")
    if "prompt_name_3" not in st.session_state:
        st.session_state.prompt_name_3 = persisted.get("prompt_name_3", "Prompt 3")

    if "openrouter_api_key" not in st.session_state:
        st.session_state.openrouter_api_key = _load_api_key_from_disk()

    with st.sidebar:
        st.header("OpenRouter")
        st.text_input(
            "API key",
            type="password",
            key="openrouter_api_key",
            placeholder="Paste your key here",
            help="Paste your OpenRouter key to run the app. It’s remembered on this computer. "
            "Or set the key in Secrets / environment instead — see the warning below if missing.",
            on_change=_save_api_key_to_disk,
        )

        api_key = _get_effective_api_key()

        st.header("Model Settings")
        if not api_key:
            st.warning(
                "No API key yet — paste above, or set OPENROUTER_API_KEY in "
                "Streamlit Secrets / .env / environment."
            )
        else:
            if (
                isinstance(st.session_state.get("openrouter_api_key"), str)
                and st.session_state.get("openrouter_api_key", "").strip()
            ):
                src = (
                    "this field (saved locally)"
                    if os.path.isfile(API_KEY_LOCAL_FILE)
                    else "this field"
                )
            elif _api_key_from_streamlit_secrets():
                src = "Streamlit Secrets"
            else:
                src = "environment / .env"
            st.success(f"API key active (from **{src}**)")
        if "evaluation_model_id" not in st.session_state:
            st.session_state.evaluation_model_id = DEFAULT_MODEL_ID
        model_id = st.selectbox(
            "Model (non-reasoning)",
            NON_REASONING_MODELS,
            key="evaluation_model_id",
            help="Full OpenRouter model id used for every image in runs below.",
        )
        st.markdown(f"**Selected model**  \n`{model_id}`")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        max_size_mb = st.slider("Image Max Size (MB)", 1.0, 8.0, 3.5, 0.5)
        timeout_s = st.number_input("Timeout (s)", min_value=30, max_value=300, value=120, step=10)
        retries = st.number_input("Retries", min_value=1, max_value=6, value=3, step=1)
        st.caption("System prompt is hard-coded in backend.")

    st.subheader("1) Upload Evaluation CSV")
    csv_file = st.file_uploader(
        "Upload CSV with columns: image_url, market_name, expected_result (ground_truth)",
        type=["csv"],
        accept_multiple_files=False,
    )
    if csv_file is not None:
        try:
            raw_df = pd.read_csv(csv_file)
            normalized_df = _normalize_csv(raw_df)
            st.session_state.csv_df = normalized_df
            st.session_state.filtered_df = normalized_df
            st.success(f"Loaded {len(normalized_df)} valid rows from CSV.")
        except Exception as exc:
            st.error(f"Failed to parse CSV: {exc}")

    if st.session_state.csv_df is not None:
        st.write(f"Total rows loaded: {len(st.session_state.csv_df)}")
        market_input = st.text_input("Market name filter (case-insensitive)", value="")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Filter by Market"):
                st.session_state.filtered_df = _filter_by_market(st.session_state.csv_df, market_input)
        with col_b:
            if st.button("Reset Filter"):
                st.session_state.filtered_df = st.session_state.csv_df

        st.write(f"Rows in current filter: {len(st.session_state.filtered_df)}")
        st.dataframe(st.session_state.filtered_df, use_container_width=True, height=250)

    st.subheader("2) Upload / Edit Up To 3 User Prompts")
    _render_evaluation_model_banner(model_id)
    for idx in (1, 2, 3):
        with st.expander(f"Prompt {idx}", expanded=(idx == 1)):
            st.text_input(
                f"Prompt {idx} Name",
                key=f"prompt_name_{idx}",
                placeholder=f"My prompt {idx}",
                on_change=_save_prompt_state_to_disk,
            )
            st.text_area(
                f"User Prompt {idx}",
                key=f"prompt_{idx}",
                height=140,
                on_change=_save_prompt_state_to_disk,
            )

    st.subheader("3) Run Prompts")
    run_col1, run_col2, run_col3, run_col4 = st.columns(4)
    run_p1 = run_col1.button("Run Prompt 1")
    run_p2 = run_col2.button("Run Prompt 2")
    run_p3 = run_col3.button("Run Prompt 3")
    run_all = run_col4.button("Run All Prompts", type="primary")

    def run_selected(prompt_indices: List[int]) -> None:
        if not api_key:
            st.error(
                "No API key — paste it in the sidebar, or set OPENROUTER_API_KEY in "
                "Streamlit Secrets / .env / environment."
            )
            return
        if st.session_state.filtered_df is None or len(st.session_state.filtered_df) == 0:
            st.error("Please upload a valid CSV (and ensure filter has rows).")
            return

        runnable: List[Dict[str, str]] = []
        skipped_names: List[str] = []
        for idx in prompt_indices:
            prompt_text = st.session_state.get(f"prompt_{idx}", "").strip()
            prompt_name = st.session_state.get(f"prompt_name_{idx}", f"Prompt {idx}").strip() or f"Prompt {idx}"
            if prompt_text:
                runnable.append({"name": prompt_name, "text": prompt_text, "slot": idx})
            else:
                skipped_names.append(prompt_name)

        if not runnable:
            st.error("No user prompt inserted. Please add at least one user prompt before running.")
            return

        if skipped_names and len(prompt_indices) > 1:
            st.info(f"Skipped empty prompts: {', '.join(skipped_names)}")

        for item in runnable:
            old_metrics: Optional[Dict[str, Any]] = None
            for existing in st.session_state.run_results:
                if existing.get("prompt_name") == item["name"]:
                    old_metrics = existing.get("metrics")
                    break
            with st.spinner(f"Running {item['name']}..."):
                result = _run_single_prompt(
                    prompt_name=item["name"],
                    prompt_slot=int(item["slot"]),
                    user_prompt=item["text"],
                    data=st.session_state.filtered_df,
                    api_key=api_key,
                    model_id=model_id,
                    temperature=temperature,
                    max_size_mb=max_size_mb,
                    timeout_s=int(timeout_s),
                    retries=int(retries),
                )
            # Upsert by prompt name: replace previous run if it exists.
            replaced = False
            for idx, existing in enumerate(st.session_state.run_results):
                if existing.get("prompt_name") == result.get("prompt_name"):
                    st.session_state.run_results[idx] = result
                    replaced = True
                    break
            if not replaced:
                st.session_state.run_results.append(result)
            st.success(_summarize_run_delta(item["name"], old_metrics, result["metrics"]))

    if run_p1:
        run_selected([1])
    if run_p2:
        run_selected([2])
    if run_p3:
        run_selected([3])
    if run_all:
        run_selected([1, 2, 3])

    pending_suggested_run = st.session_state.get("pending_suggested_run")
    if pending_suggested_run:
        st.session_state.pending_suggested_run = None
        if not api_key:
            st.error(
                "No API key — paste it in the sidebar, or set OPENROUTER_API_KEY in "
                "Streamlit Secrets / .env / environment."
            )
        elif st.session_state.filtered_df is None or len(st.session_state.filtered_df) == 0:
            st.error("Please upload a valid CSV (and ensure filter has rows).")
        else:
            pn = str(pending_suggested_run["prompt_name"])
            old_metrics: Optional[Dict[str, Any]] = None
            for existing in st.session_state.run_results:
                if existing.get("prompt_name") == pn:
                    old_metrics = existing.get("metrics")
                    break
            with st.spinner(f"Running revised prompt: {pn}..."):
                revised_result = _run_single_prompt(
                    prompt_name=pn,
                    prompt_slot=int(pending_suggested_run["prompt_slot"]),
                    user_prompt=str(pending_suggested_run["prompt_text"]),
                    data=st.session_state.filtered_df,
                    api_key=api_key,
                    model_id=model_id,
                    temperature=temperature,
                    max_size_mb=max_size_mb,
                    timeout_s=int(timeout_s),
                    retries=int(retries),
                )
            replaced = False
            for idx, existing in enumerate(st.session_state.run_results):
                if existing.get("prompt_name") == revised_result.get("prompt_name"):
                    st.session_state.run_results[idx] = revised_result
                    replaced = True
                    break
            if not replaced:
                st.session_state.run_results.append(revised_result)
            st.success(_summarize_run_delta(pn, old_metrics, revised_result["metrics"]))

    st.subheader("Run Results (Per Prompt)")
    if not st.session_state.run_results:
        st.info("No runs yet.")
    else:
        st.markdown(
            """
            ### Metric Calculations
            **Accuracy:** % of images where the model’s prediction **matches** ground truth:  
            **(TP + TN) ÷ total images × 100%**.  
            - **TP** = ground truth **Pass** and model **Pass**  
            - **TN** = ground truth **Fail** and model **Fail**  
            - **Total** = number of images in the run

            **False Positive (GP Leakage):** % of **Fail** wrongly marked **Pass** by AI.  
            **False Negative (UX Friction):** % of **Pass** wrongly marked **Fail** by AI.

            **Costs (USD, estimated from OpenRouter `usage`):** **Vision** = image classification only. **Text APIs** = automated insights + prompt optimizer (and any failed model fallbacks in those flows). **Total est.** = vision + text.
            """
        )

        summary_rows = []
        for r in st.session_state.run_results:
            m = r["metrics"]
            vision_c = float(m.get("cost_usd", 0.0))
            aux_c = float(r.get("aux_cost_usd", 0.0))
            summary_rows.append(
                {
                    "Prompt name": r["prompt_name"],
                    "Vision model (OpenRouter)": r.get("model_id", DEFAULT_MODEL_ID),
                    "Accuracy %": m["accuracy"],
                    "False positive rate % (GP leakage)": m["fpr"],
                    "False negative rate % (UX friction)": m["fnr"],
                    "# False positives (bad Pass)": m["fp"],
                    "# False negatives (bad Fail)": m["fn"],
                    "# Ground truth Pass (in eval set)": m["gt_pass_count"],
                    "# Ground truth Fail (in eval set)": m["gt_fail_count"],
                    "# Images evaluated": m["total_images"],
                    "Markets in dataset": m["markets"],
                    "Vision eval cost est. (USD)": round(vision_c, 4),
                    "Insights + optimizer cost est. (USD)": round(aux_c, 4),
                    "Total est. cost (USD)": round(vision_c + aux_c, 4),
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("### Per-Run FP/FN Explorer")
        for run_idx, run_result in enumerate(st.session_state.run_results, start=1):
            run_name = run_result["prompt_name"]
            run_model = run_result.get("model_id", DEFAULT_MODEL_ID)
            run_metrics = run_result["metrics"]
            all_rows = run_result["rows"]
            fp_rows = [r for r in all_rows if r["ground_truth"] == "FAIL" and r["pred"] == "PASS"]
            fn_rows = [r for r in all_rows if r["ground_truth"] == "PASS" and r["pred"] == "FAIL"]
            n_fp = len(fp_rows)
            n_fn = len(fn_rows)
            n_mistakes = n_fp + n_fn
            run_aux = float(run_result.get("aux_cost_usd", 0.0))

            with st.expander(
                f"Run {run_idx}: {run_name} | Model={run_model} | FP={n_fp} | FN={n_fn}",
                expanded=(run_idx == len(st.session_state.run_results)),
            ):
                # LTR: All mistakes | GP | UX — "All mistakes" is the leftmost (first) tab.
                all_failed_tab, fp_tab, fn_tab = st.tabs(
                    [
                        f"All mistakes — {n_mistakes}",
                        f"GP leakage (FP) — {n_fp}",
                        f"UX friction (FN) — {n_fn}",
                    ]
                )
                err_by_mkt = _compute_market_error_breakdown(all_rows)
                err_fp_mkt = _compute_market_error_breakdown_fp_only(all_rows)
                err_fn_mkt = _compute_market_error_breakdown_fn_only(all_rows)

                with all_failed_tab:
                    _show_run_metric_cards(run_metrics, focus="all", aux_cost_usd=run_aux)
                    has_failed = bool(_failed_rows(all_rows))

                    st.markdown("#### Insights (full run — FP & FN)")
                    st.caption(
                        "**Main failure points** → headline + reason + **(N)** images for that theme (from model reason counts) · **Action 1** → GP leakage (FP) · "
                        "**Action 2** → UX friction (FN) · **Action 3** → **one** synthesized “most balanced” user-prompt stance "
                        "(not a mash-up of 1+2). **GP** / **UX** tabs are single-side only."
                    )
                    if not has_failed:
                        st.success("No errors in this run — nothing to diagnose.")
                    elif err_by_mkt.empty:
                        st.caption("No per-market breakdown for these errors.")
                    else:
                        st.caption("Errors by market — FP and FN combined (context for insights below)")
                        st.dataframe(err_by_mkt, use_container_width=True, hide_index=True)

                    _render_insights_panel(
                        focus="all",
                        run_idx=run_idx,
                        run_metrics=run_metrics,
                        run_result=run_result,
                        run_model=run_model,
                        market_breakdown_md=err_by_mkt.to_string(index=False) if not err_by_mkt.empty else "",
                        fp_rows=fp_rows,
                        fn_rows=fn_rows,
                        api_key=api_key,
                        model_id=model_id,
                        temperature=float(temperature),
                        timeout_s=int(timeout_s),
                        retries=int(retries),
                        should_run=bool(fp_rows or fn_rows),
                    )

                    if has_failed:
                        _show_failed_photos_table(all_rows)
                        _show_failed_gallery(all_rows, f"run_{run_idx}")

                with fp_tab:
                    _show_run_metric_cards(run_metrics, focus="fp", aux_cost_usd=run_aux)

                    st.markdown("#### Insights (GP leakage)")
                    if not fp_rows:
                        st.caption("No false positives — no GP-specific insights.")
                    else:
                        st.caption(
                            "**Main failure points** → headline + reason + **(N)** (GP leakage counts) · then **Actions 1–3**."
                        )
                        _render_insights_panel(
                            focus="fp",
                            run_idx=run_idx,
                            run_metrics=run_metrics,
                            run_result=run_result,
                            run_model=run_model,
                            market_breakdown_md=err_fp_mkt.to_string(index=False)
                            if not err_fp_mkt.empty
                            else "",
                            fp_rows=fp_rows,
                            fn_rows=fn_rows,
                            api_key=api_key,
                            model_id=model_id,
                            temperature=float(temperature),
                            timeout_s=int(timeout_s),
                            retries=int(retries),
                            should_run=True,
                        )

                    if err_fp_mkt.empty:
                        st.caption("No GP leakage by market (no false positives).")
                    else:
                        st.caption("GP leakage by market (false positives only)")
                        st.dataframe(err_fp_mkt, use_container_width=True, hide_index=True)
                    _show_error_bucket(
                        fp_rows,
                        "GP leakage — model said Pass but ground truth is Fail",
                        run_key=f"run_{run_idx}_fp",
                    )

                with fn_tab:
                    _show_run_metric_cards(run_metrics, focus="fn", aux_cost_usd=run_aux)

                    st.markdown("#### Insights (UX friction)")
                    if not fn_rows:
                        st.caption("No false negatives — no UX-specific insights.")
                    else:
                        st.caption(
                            "**Main failure points** → headline + reason + **(N)** (UX friction counts) · then **Actions 1–3**."
                        )
                        _render_insights_panel(
                            focus="fn",
                            run_idx=run_idx,
                            run_metrics=run_metrics,
                            run_result=run_result,
                            run_model=run_model,
                            market_breakdown_md=err_fn_mkt.to_string(index=False)
                            if not err_fn_mkt.empty
                            else "",
                            fp_rows=fp_rows,
                            fn_rows=fn_rows,
                            api_key=api_key,
                            model_id=model_id,
                            temperature=float(temperature),
                            timeout_s=int(timeout_s),
                            retries=int(retries),
                            should_run=True,
                        )

                    if err_fn_mkt.empty:
                        st.caption("No UX friction by market (no false negatives).")
                    else:
                        st.caption("UX friction by market (false negatives only)")
                        st.dataframe(err_fn_mkt, use_container_width=True, hide_index=True)
                    _show_error_bucket(
                        fn_rows,
                        "UX friction — model said Fail but ground truth is Pass",
                        run_key=f"run_{run_idx}_fn",
                    )

        # Optimization helper based on latest run errors
        latest = st.session_state.run_results[-1]
        latest_rows = latest["rows"]
        latest_fp_rows = [r for r in latest_rows if r["ground_truth"] == "FAIL" and r["pred"] == "PASS"]
        latest_fn_rows = [r for r in latest_rows if r["ground_truth"] == "PASS" and r["pred"] == "FAIL"]
        latest_prompt_text = latest.get("prompt_text", "")
        latest_fpr = float(latest.get("metrics", {}).get("fpr", 0.0))
        latest_fnr = float(latest.get("metrics", {}).get("fnr", 0.0))

        if latest_fpr > latest_fnr:
            default_objective = "Minimize False Positives (GP Leakage)"
        elif latest_fnr > latest_fpr:
            default_objective = "Minimize False Negatives (UX Friction)"
        else:
            default_objective = "Balanced"

        objective_options = [
            "Minimize False Positives (GP Leakage)",
            "Minimize False Negatives (UX Friction)",
            "Balanced",
        ]
        default_idx = objective_options.index(default_objective)
        if "optimize_objective_select" not in st.session_state:
            st.session_state.optimize_objective_select = objective_options[default_idx]

        st.markdown("### Optimize Prompt From Latest Run Errors")
        st.caption(
            f"Optimizer model: `{OPTIMIZER_MODEL_ID}`. Vision eval uses **the model you choose** in the sidebar "
            f"(default `{DEFAULT_MODEL_ID}`). Eval is **always non-reasoning** (no extended chain-of-thought). "
            "The optimizer is told to tailor the user prompt for your selected model. Uses the same **Insights** "
            "as Per-Run Explorer when available."
        )
        st.warning(
            "**Balance:** Tightening rules to cut FP can spike FN (and the opposite). "
            "Apply suggestions in **small steps**, re-run eval after each change, and use **Balanced** objective "
            "when both metrics matter — the optimizer is instructed not to wreck one side while fixing the other."
        )
        objective = st.selectbox(
            "What should we optimize for?",
            objective_options,
            key="optimize_objective_select",
        )

        if st.button("Optimize Latest Prompt", type="primary"):
            if not latest_prompt_text.strip():
                st.error("Latest run does not have prompt text to optimize.")
            else:
                with st.spinner("Refreshing insights (if needed) and generating optimized prompt..."):
                    try:
                        latest_run_idx = len(st.session_state.run_results)
                        latest_run_model = latest.get("model_id", DEFAULT_MODEL_ID)
                        latest_metrics = latest["metrics"]
                        err_latest = _compute_market_error_breakdown(latest_rows)
                        err_latest_fp = _compute_market_error_breakdown_fp_only(latest_rows)
                        err_latest_fn = _compute_market_error_breakdown_fn_only(latest_rows)
                        if api_key:
                            _ensure_insights_for_optimizer(
                                latest_run_idx,
                                latest_metrics,
                                latest,
                                latest_run_model,
                                latest_fp_rows,
                                latest_fn_rows,
                                err_latest,
                                err_latest_fp,
                                err_latest_fn,
                                api_key,
                                model_id,
                                float(temperature),
                                int(timeout_s),
                                int(retries),
                            )
                        insights_bundle = _format_insights_for_optimizer_prompt(objective, latest_run_idx)
                        suggestion = _suggest_prompt_from_errors(
                            api_key=api_key,
                            optimizer_model_id=OPTIMIZER_MODEL_ID,
                            runtime_model_id=model_id,
                            current_prompt=latest_prompt_text,
                            fp_rows=latest_fp_rows,
                            fn_rows=latest_fn_rows,
                            objective=objective,
                            user_goal_note="",
                            timeout_s=int(timeout_s),
                            retries=int(retries),
                            insights_markdown=insights_bundle,
                            run_metrics=latest_metrics,
                        )
                        suggestion_text, used_model, opt_cost = suggestion
                        _add_aux_cost_to_run(latest_run_idx, opt_cost)
                        st.session_state.optimized_prompt_source = latest["prompt_name"]
                        st.session_state.optimized_prompt_suggestion = suggestion_text
                        st.session_state.optimized_prompt_baseline = latest_prompt_text
                        st.session_state.optimized_prompt_model_used = used_model
                        st.session_state.apply_suggest_target_slot = "Prompt 1"
                        _apply_optimizer_result_to_prompt_slots(suggestion_text, latest, latest_prompt_text)
                        st.success(
                            "Saved **revised** prompt to **Prompt 1** and **original** (pre-optimization) to **Prompt 2**. "
                            "If Prompt 1 had unsynced edits vs the last run, the old text is in **Prompt 3**."
                        )
                    except Exception as exc:
                        st.error(f"Optimizer failed: {exc}")

        if st.session_state.optimized_prompt_suggestion:
            st.info(
                f"Latest suggestion from run **{st.session_state.optimized_prompt_source}** — "
                "already saved to **Prompt 1** / **Prompt 2** when you ran the optimizer. "
                "Use **Apply** to copy to another slot or queue a run."
            )
            if st.session_state.optimized_prompt_model_used:
                st.caption(f"Optimizer model used: `{st.session_state.optimized_prompt_model_used}`")
            st.text_area(
                "Suggested Prompt",
                value=st.session_state.optimized_prompt_suggestion,
                height=220,
                key="suggested_prompt_view",
            )
            if "apply_suggest_target_slot" not in st.session_state:
                st.session_state.apply_suggest_target_slot = "Prompt 1"
            target_prompt_slot = st.selectbox(
                "Apply suggestion to",
                options=["Prompt 1", "Prompt 2", "Prompt 3"],
                key="apply_suggest_target_slot",
            )
            st.button(
                "Apply Suggestion (Run Now)",
                key="apply_suggest_btn",
                on_click=_apply_suggestion_to_prompt,
            )
            if st.session_state.apply_suggest_success:
                st.success(st.session_state.apply_suggest_success)
                st.session_state.apply_suggest_success = ""


if __name__ == "__main__":
    main()

