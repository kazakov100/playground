"""
Batch evaluation helpers (sequential). Not used by `app.py`, which runs its own parallel loop.
Kept for reuse or older workflows; safe to import `run_optimizer` from notebooks/tests.
"""

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .openrouter import classify_image


@dataclass
class ImageItem:
    file: str
    gt: str
    source: str = ""


def infer_label(filename: str) -> str:
    name = os.path.basename(filename).upper()
    if name.startswith("PASS"):
        return "PASS"
    if name.startswith("FAIL"):
        return "FAIL"
    return ""


def load_market_photos_from_csv(market: str, csv_path: str = "ai photo review - examples.csv") -> List[ImageItem]:
    if not market:
        return []
    if not os.path.exists(csv_path):
        return []

    df = pd.read_csv(csv_path)
    expected = {"market", "image_url", "evaluation_status"}
    if not expected.issubset(set(df.columns)):
        return []

    m = market.strip().lower()
    filtered = df[df["market"].astype(str).str.strip().str.lower() == m]
    items: List[ImageItem] = []
    for _, row in filtered.iterrows():
        status = str(row["evaluation_status"]).upper().strip()
        if status not in ("PASS", "FAIL"):
            continue
        image_url = str(row["image_url"]).strip()
        if not image_url:
            continue
        items.append(ImageItem(file=image_url, gt=status, source="csv"))
    return items


def build_items(uploaded_paths: List[str], market: str, csv_path: str = "ai photo review - examples.csv") -> List[ImageItem]:
    items: List[ImageItem] = []
    for p in uploaded_paths:
        gt = infer_label(p)
        if gt:
            items.append(ImageItem(file=p, gt=gt, source="upload"))

    items.extend(load_market_photos_from_csv(market, csv_path=csv_path))
    return items


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    total = len(rows)
    correct = sum(1 for r in rows if r["gt"] == r["pred"])
    fp = sum(1 for r in rows if r["gt"] == "FAIL" and r["pred"] == "PASS")
    fn = sum(1 for r in rows if r["gt"] == "PASS" and r["pred"] == "FAIL")
    tn = sum(1 for r in rows if r["gt"] == "FAIL" and r["pred"] == "FAIL")
    tp = sum(1 for r in rows if r["gt"] == "PASS" and r["pred"] == "PASS")

    accuracy = (correct / total * 100.0) if total else 0.0
    fpr = (fp / (fp + tn) * 100.0) if (fp + tn) else 0.0
    fnr = (fn / (fn + tp) * 100.0) if (fn + tp) else 0.0

    return {
        "accuracy": round(accuracy, 2),
        "fpr": round(fpr, 2),
        "fnr": round(fnr, 2),
        "photos_tested": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_optimizer(
    items: List[ImageItem],
    api_key: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    iterations: int = 1,
    temperature: float = 0.1,
    timeout_s: int = 120,
    retries: int = 3,
    max_size_mb: float = 3.5,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Runs evaluation history loop. For now this evaluates the same prompt across iterations
    to preserve predictable behavior and avoid drift, while still keeping history scaffolding.
    """
    history: List[Dict[str, Any]] = []
    all_rows_latest: List[Dict[str, Any]] = []
    total_cost = 0.0

    if iterations < 1:
        iterations = 1

    for iteration in range(1, iterations + 1):
        rows: List[Dict[str, Any]] = []
        total = len(items)
        for idx, item in enumerate(items, start=1):
            if progress_cb:
                progress_cb(idx, total, f"Iteration {iteration}: Processing... ({idx}/{total})")

            def status(message: str) -> None:
                if progress_cb:
                    progress_cb(idx, total, f"Iteration {iteration}: {message} ({idx}/{total})")

            try:
                out, item_cost = classify_image(
                    api_key=api_key,
                    model_id=model_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_path_or_url=item.file,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    retries=retries,
                    max_size_mb=max_size_mb,
                    progress_cb=status,
                )
                pred = out.get("decision", "FAIL")
                reason = out.get("reason", "")
                visual_checklist = out.get("visual_checklist", {}) or {}
                total_cost += item_cost
            except Exception as exc:
                pred = "FAIL"
                reason = f"ERROR: {str(exc)[:180]}"
                visual_checklist = {}

            rows.append(
                {
                    "file": item.file,
                    "gt": item.gt,
                    "pred": pred,
                    "reason": reason,
                    "visual_checklist": visual_checklist,
                }
            )

        metrics = compute_metrics(rows)
        history.append({"iteration": iteration, "metrics": metrics, "rows": rows})
        all_rows_latest = rows

    failed_rows = [r for r in all_rows_latest if r["gt"] != r["pred"]]
    fp_rows = [r for r in all_rows_latest if r["gt"] == "FAIL" and r["pred"] == "PASS"]
    fn_rows = [r for r in all_rows_latest if r["gt"] == "PASS" and r["pred"] == "FAIL"]

    history_df = pd.DataFrame(
        [
            {
                "iteration": h["iteration"],
                "accuracy": h["metrics"]["accuracy"],
                "false_positive_rate": h["metrics"]["fpr"],
                "false_negative_rate": h["metrics"]["fnr"],
                "photos_tested": h["metrics"]["photos_tested"],
            }
            for h in history
        ]
    )

    return {
        "history": history,
        "history_df": history_df,
        "latest_rows": all_rows_latest,
        "failed_rows": failed_rows,
        "fp_rows": fp_rows,
        "fn_rows": fn_rows,
        "metrics": history[-1]["metrics"] if history else compute_metrics([]),
        "cost_usd": round(total_cost, 4),
    }

