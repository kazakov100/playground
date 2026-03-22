import os
import json
from typing import Any, Dict, List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st

from core.openrouter import classify_image, post_with_retries


DEFAULT_MODEL_ID = "anthropic/claude-sonnet-4.6"
NON_REASONING_MODELS = [
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-sonnet-4",
]
# Use a cheaper text model for prompt optimization (no vision needed).
OPTIMIZER_MODEL_ID = "openai/gpt-4o-mini"

PROMPT_STATE_FILE = os.path.join(os.path.dirname(__file__), ".prompt_state.json")

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

        h1, h2, h3, h4, h5, h6, p, label, span, div, .stMarkdown, .stCaption {
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
        .stNumberInput input,
        div[data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.96) !important;
            color: var(--ui-text) !important;
            border: 1px solid var(--ui-border) !important;
            border-radius: 12px !important;
            padding: 0.55rem 0.75rem !important;
            line-height: 1.4 !important;
            -webkit-text-fill-color: var(--ui-text) !important;
            caret-color: var(--ui-text) !important;
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

        /* Left sidebar readability */
        [data-testid="stSidebar"] {
            background: rgba(255,255,255,0.86) !important;
            border-right: 1px solid var(--ui-border) !important;
        }
        [data-testid="stSidebar"] * {
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


def _apply_suggestion_to_prompt() -> None:
    """Safe callback: apply optimized suggestion and queue immediate run."""
    suggestion = st.session_state.get("optimized_prompt_suggestion", "").strip()
    if not suggestion:
        st.session_state["apply_suggest_success"] = ""
        return

    target_prompt_slot = st.session_state.get("apply_suggest_target_slot", "Prompt 1")
    slot_idx = {"Prompt 1": 1, "Prompt 2": 2, "Prompt 3": 3}.get(target_prompt_slot, 1)
    source_name = st.session_state.get("optimized_prompt_source", f"Prompt {slot_idx}").strip() or f"Prompt {slot_idx}"
    revised_name = source_name if source_name.lower().endswith(" - revised") else f"{source_name} - revised"

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
) -> tuple[str, str]:
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing.")

    objective_instructions = {
        "Minimize False Positives (GP Leakage)": "Prioritize reducing false positives (GT FAIL predicted PASS).",
        "Minimize False Negatives (UX Friction)": "Prioritize reducing false negatives (GT PASS predicted FAIL).",
        "Balanced": "Balance reducing both false positives and false negatives.",
    }
    objective_text = objective_instructions.get(objective, objective_instructions["Balanced"])

    fp_lines = _build_error_lines(fp_rows)
    fn_lines = _build_error_lines(fn_rows)
    max_len = max(300, int(len(current_prompt) * 1.2))

    prompt = f"""
You are an expert prompt engineer for image classification policy prompts.
Generate ONE revised user prompt.

Objective:
{objective_text}

Additional user guidance:
{user_goal_note or "None"}

Current prompt:
{current_prompt}

Observed False Positives (GT FAIL but predicted PASS):
{fp_lines}

Observed False Negatives (GT PASS but predicted FAIL):
{fn_lines}

Guardrails:
1) Keep the revised prompt concise and production-ready.
2) Preserve original intent; only fix error-prone ambiguity.
3) Do not add long examples, JSON schemas, or extra meta-instructions.
4) Keep output length <= {max_len} characters.
5) Output ONLY the revised prompt text, no commentary.
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
    for candidate_model in model_candidates:
        payload = {
            "model": candidate_model,
            "temperature": 0.1,
            # Force compatible providers to avoid provider-policy mismatch.
            # Keep provider preferences broad; account/provider policy may override this.
            "provider": {"allow_fallbacks": True},
            "messages": [
                {"role": "system", "content": "You improve prompts without changing product intent."},
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
            choices = data.get("choices", [])
            if not choices:
                last_err = f"Invalid optimizer response for model {candidate_model}: {data}"
                continue
            suggestion = (choices[0]["message"]["content"] or "").strip()
            if not suggestion:
                last_err = f"Empty optimizer output for model {candidate_model}"
                continue
            return suggestion, candidate_model
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
        except Exception as exc:
            cost = 0.0
            pred = "FAIL"
            reason = f"ERROR: {str(exc)[:180]}"

        return {
            "image_url": image_url,
            "market_name": market_name,
            "ground_truth": gt,
            "pred": pred,
            "reason": reason,
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
    }


def _show_run_metric_cards(metrics: Dict[str, Any]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2f}%")
    c2.metric("False Positive Rate", f"{metrics.get('fpr', 0):.2f}%")
    c3.metric("False Negative Rate", f"{metrics.get('fnr', 0):.2f}%")
    c4.metric("Cost", f"${metrics.get('cost_usd', 0):.4f}")
    c5.metric("Photos Tested", str(metrics.get("total_images", 0)))


def _show_failed_gallery(rows: List[Dict[str, Any]]) -> None:
    failed = [r for r in rows if r["ground_truth"] != r["pred"]]
    if not failed:
        st.info("No failed images for this run.")
        return
    cols = st.columns(4)
    for i, item in enumerate(failed):
        with cols[i % 4]:
            st.image(item["image_url"], use_container_width=True)
            st.caption(f"Market: {item['market_name']} | GT: {item['ground_truth']} | Pred: {item['pred']}")
            st.write(item.get("reason", ""))


def _show_error_bucket(rows: List[Dict[str, Any]], title: str, run_key: str) -> None:
    st.markdown(f"#### {title} ({len(rows)})")
    if not rows:
        st.info(f"No {title.lower()} in this run.")
        return

    table_rows = [
        {
            "image_url": r["image_url"],
            "market_name": r["market_name"],
            "ai_judgement": r["pred"],
            "ground_truth": r["ground_truth"],
            "ai_review": r.get("reason", ""),
        }
        for r in rows
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, height=220)

    cols = st.columns(4)
    for idx, item in enumerate(rows):
        with cols[idx % 4]:
            st.markdown(
                (
                    f"**AI:** `{item['pred']}`  \n"
                    f"**GT:** `{item['ground_truth']}`  \n"
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

    api_key = _load_openrouter_key()

    with st.sidebar:
        st.header("Model Settings")
        if not api_key:
            st.error("OPENROUTER_API_KEY not found in environment or .env")
        model_id = st.selectbox(
            "Model (non-reasoning)",
            NON_REASONING_MODELS,
            index=0,
        )
        st.caption(f"Active model: `{model_id}`")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        max_size_mb = st.slider("Image Max Size (MB)", 1.0, 8.0, 3.5, 0.5)
        timeout_s = st.number_input("Timeout (s)", min_value=30, max_value=300, value=120, step=10)
        retries = st.number_input("Retries", min_value=1, max_value=6, value=3, step=1)
        st.caption("System prompt is hard-coded in backend.")

    st.info(f"Current selected model: {model_id}")

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
            st.error("OPENROUTER_API_KEY not found. Please add it to your .env.")
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
            st.error("OPENROUTER_API_KEY not found. Please add it to your .env.")
        elif st.session_state.filtered_df is None or len(st.session_state.filtered_df) == 0:
            st.error("Please upload a valid CSV (and ensure filter has rows).")
        else:
            with st.spinner(f"Running revised prompt: {pending_suggested_run['prompt_name']}..."):
                revised_result = _run_single_prompt(
                    prompt_name=str(pending_suggested_run["prompt_name"]),
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
            st.success(f"Finished revised run: {revised_result['prompt_name']}")

    st.subheader("Run Results (Per Prompt)")
    if not st.session_state.run_results:
        st.info("No runs yet.")
    else:
        st.markdown(
            """
            ### Metric Calculations
            **False Positive (GP Leakage):** % of **Fail** wrongly marked **Pass** by AI.  
            **False Negative (UX Friction):** % of **Pass** wrongly marked **Fail** by AI.
            """
        )

        summary_rows = []
        for r in st.session_state.run_results:
            m = r["metrics"]
            summary_rows.append(
                {
                    "prompt": r["prompt_name"],
                    "model": r.get("model_id", DEFAULT_MODEL_ID),
                    "false_positive_gp_leakage_pct": m["fpr"],
                    "false_negative_ux_friction_pct": m["fnr"],
                    "false_positive_count": m["fp"],
                    "false_negative_count": m["fn"],
                    "gt_pass_count": m["gt_pass_count"],
                    "gt_fail_count": m["gt_fail_count"],
                    "total_images_evaluated": m["total_images"],
                    "market_name(s)": m["markets"],
                    "cost_usd": m["cost_usd"],
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

        st.markdown("### Per-Run FP/FN Explorer")
        for run_idx, run_result in enumerate(st.session_state.run_results, start=1):
            run_name = run_result["prompt_name"]
            run_model = run_result.get("model_id", DEFAULT_MODEL_ID)
            run_metrics = run_result["metrics"]
            all_rows = run_result["rows"]
            fp_rows = [r for r in all_rows if r["ground_truth"] == "FAIL" and r["pred"] == "PASS"]
            fn_rows = [r for r in all_rows if r["ground_truth"] == "PASS" and r["pred"] == "FAIL"]

            with st.expander(
                f"Run {run_idx}: {run_name} | Model={run_model} | FP={len(fp_rows)} | FN={len(fn_rows)}",
                expanded=(run_idx == len(st.session_state.run_results)),
            ):
                _show_run_metric_cards(run_metrics)
                fp_tab, fn_tab, all_failed_tab = st.tabs(
                    ["False Positives", "False Negatives", "All Failed"]
                )
                with fp_tab:
                    _show_error_bucket(fp_rows, "False Positives", run_key=f"run_{run_idx}_fp")
                with fn_tab:
                    _show_error_bucket(fn_rows, "False Negatives", run_key=f"run_{run_idx}_fn")
                with all_failed_tab:
                    _show_failed_gallery(all_rows)

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

        st.markdown("### Optimize Prompt From Latest Run Errors")
        st.caption(f"Optimizer model: `{OPTIMIZER_MODEL_ID}`")
        objective = st.selectbox(
            "What should we optimize for?",
            objective_options,
            index=default_idx,
            key="optimize_objective_select",
        )
        st.info(f"Current optimization target: **{objective}**")

        if st.button("Optimize Latest Prompt", type="primary"):
            if not latest_prompt_text.strip():
                st.error("Latest run does not have prompt text to optimize.")
            else:
                with st.spinner("Generating optimized prompt suggestion..."):
                    try:
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
                        )
                        suggestion_text, used_model = suggestion
                        st.session_state.optimized_prompt_source = latest["prompt_name"]
                        st.session_state.optimized_prompt_suggestion = suggestion_text
                        st.session_state.optimized_prompt_model_used = used_model
                        source_slot = int(latest.get("prompt_slot", 1))
                        st.session_state.apply_suggest_target_slot = f"Prompt {source_slot}"
                    except Exception as exc:
                        st.error(f"Optimizer failed: {exc}")

        if st.session_state.optimized_prompt_suggestion:
            st.info(
                f"Suggestion generated from run: {st.session_state.optimized_prompt_source}. "
                "Use apply to save and run it immediately."
            )
            if st.session_state.optimized_prompt_model_used:
                st.caption(f"Optimizer model used: `{st.session_state.optimized_prompt_model_used}`")
            st.text_area(
                "Suggested Prompt",
                value=st.session_state.optimized_prompt_suggestion,
                height=220,
                key="suggested_prompt_view",
            )
            target_prompt_slot = st.selectbox(
                "Apply suggestion to",
                options=["Prompt 1", "Prompt 2", "Prompt 3"],
                index=["Prompt 1", "Prompt 2", "Prompt 3"].index(
                    st.session_state.get("apply_suggest_target_slot", "Prompt 1")
                    if st.session_state.get("apply_suggest_target_slot", "Prompt 1") in ["Prompt 1", "Prompt 2", "Prompt 3"]
                    else "Prompt 1"
                ),
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

