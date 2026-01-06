"""
Streamlit UI for Prompt Optimizer
Run with: streamlit run app_streamlit.py

This is a complete Streamlit migration from Gradio.
Streamlit handles file uploads much better than Gradio.
"""

import streamlit as st
import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO
import base64
import mimetypes
import json
import logging
import threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from PIL import Image

# =========================================================
# CONFIGURATION (from notebook cell 0)
# =========================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL_ID = "anthropic/claude-sonnet-4"
MODEL_TEMPERATURE = 0.2
MIN_PASS_IMAGES_REQUIRED = 3
OPTIMIZATION_ITERATIONS = 10
TARGET_RECALL = 95.0
TARGET_PRECISION = 95.0
EARLY_STOP_THRESHOLD = 95.0
OPENROUTER_TIMEOUT_SECONDS = 120
OPENROUTER_RETRIES = 3
MAX_WORKERS = 5

BASE_RULES = [
    "It is enough that one of the Bird vehicles in the picture is PASS to grant a PASS for the image.",
    "Evaluate only the two vehicle types used in this market:\n • Bird scooters (Gray-and-blue)\n • Spin scooters (orange)"
]

DEFAULT_SYSTEM_PROMPT = """
You are a micromobility parking enforcement officer.
Analyze this parking photo of a shared e-scooter or bicycle.
Rate it as PASS or FAIL and provide feedback accordingly.

Condense your feedback into a single, short sentence.
If FAIL, make it actionable so the rider knows what they need to improve.
""".strip()

BASE_RULES_TEXT = "\n\n".join([f"• {rule}" for rule in BASE_RULES]) if BASE_RULES else ""

DEFAULT_BASE_USER_PROMPT = f"""
BASE RULES (must always be followed):
{BASE_RULES_TEXT}

Apply the following parking rules:

1. The vehicle must be fully visible in the photo. It cannot be cut off. The photo should include enough surrounding space to judge whether the vehicle is parked appropriately.
2. Vehicles must not block sidewalks or pedestrian paths.
3. Parking on sidewalks is only ok if the vehicle is fully to the side and does not obstruct pedestrian flow.
4. Parking in the furniture zone of the sidewalk/street is ok and better than if parked in the middle of the sidewalk
5. Parking in designated micromobility spaces is preferred, but optional. Designated parking spaces are usually marked with paint on the ground or clear sign posts or other signage.
6. Vehicles must not block streets, driveways, public transport stations or entrances
7. Parking near crosswalks, ramps, or handicap-accessible areas is not allowed.
8. The vehicle must be upright and not tipped over.

Only refer to the brand of the scooter or bike if you are absolutely sure you detected the brand correctly (Bird, Spin).
""".strip()

# =========================================================
# LOGGING SETUP
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global stop event
stop_event = threading.Event()

# =========================================================
# CORE FUNCTIONS (copied from notebook)
# =========================================================

def post_with_retries(
    url: str,
    headers: dict,
    payload: dict,
    timeout_s: int = OPENROUTER_TIMEOUT_SECONDS,
    retries: int = OPENROUTER_RETRIES,
) -> dict:
    """POST with retries + exponential backoff."""
    logger.info(f"Making request to {url} (timeout={timeout_s}s, retries={retries})")
    last_err: Optional[str] = None
    for attempt in range(1, retries + 1):
        try:
            logger.debug(f"Attempt {attempt}/{retries}")
            r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
            if r.status_code == 200:
                data = r.json()
                if "error" in data:
                    error_info = data.get("error", {})
                    error_msg = error_info.get("message", "Unknown error")
                    if "metadata" in error_info and "raw" in error_info["metadata"]:
                        try:
                            raw_error = json.loads(error_info["metadata"]["raw"])
                            if "error" in raw_error and "message" in raw_error["error"]:
                                error_msg = raw_error["error"]["message"]
                        except:
                            pass
                    last_err = f"Provider error: {error_msg}"
                    logger.warning(f"Request returned error in response: {last_err}")
                else:
                    logger.info(f"Request successful on attempt {attempt}")
                    return data
            else:
                last_err = f"HTTP {r.status_code}: {r.text[:600]}"
                logger.warning(f"Request failed with status {r.status_code}: {last_err}")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, BrokenPipeError) as e:
            last_err = f"Connection error: {type(e).__name__}: {str(e)}"
            logger.warning(f"Request connection error on attempt {attempt}: {e}")
        except Exception as e:
            last_err = str(e)
            logger.warning(f"Request exception on attempt {attempt}: {type(e).__name__}: {e}")

        if attempt < retries:
            wait_time = 2 ** (attempt - 1)
            logger.info(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)

    error_msg = f"OpenRouter request failed after {retries} attempts. Last error: {last_err}"
    logger.error(error_msg)
    raise RuntimeError(error_msg)

def load_base_photos_from_csv(market: str, csv_path: str = "ai photo review - examples.csv") -> Tuple[List[Dict[str, Any]], str]:
    """Load base photos from CSV file filtered by market (case-insensitive)."""
    if not market or not market.strip():
        return [], ""
    
    market = market.strip()
    
    # Simplified: Only check current directory and one level up
    possible_paths = [
        csv_path,  # Current directory
        os.path.join("..", csv_path),  # Parent directory (one level up)
    ]
    
    csv_file_path = None
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and os.path.isfile(abs_path):
            csv_file_path = abs_path
            break
    
    if not csv_file_path:
        return [], f"CSV file not found: {csv_path}. Please ensure the file exists in the project directory."
    
    try:
        df = pd.read_csv(csv_file_path)
        logger.info(f"Loaded CSV file: {csv_file_path} ({len(df)} rows)")
        
        required_cols = ['market', 'image_url', 'evaluation_status', 'evaluation_message']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return [], f"CSV file missing required columns: {', '.join(missing_cols)}. Found columns: {', '.join(df.columns)}"
        
        df_filtered = df[df['market'].astype(str).str.upper().str.strip() == market.upper().strip()]
        
        if len(df_filtered) == 0:
            unique_markets = df['market'].astype(str).str.strip().unique()[:10]
            return [], f"Market '{market}' not found in CSV file. Available markets (sample): {', '.join(unique_markets)}"
        
        df_filtered = df_filtered[df_filtered['image_url'].notna() & (df_filtered['image_url'].astype(str).str.strip() != "")]
        
        if len(df_filtered) == 0:
            return [], f"Market '{market}' found but has no photos with valid image_url"
        
        base_photos = []
        for _, row in df_filtered.iterrows():
            status = str(row['evaluation_status']).strip().upper()
            if status not in ['PASS', 'FAIL']:
                logger.warning(f"Skipping row with invalid evaluation_status: {status}")
                continue
            
            base_photos.append({
                'image_url': str(row['image_url']).strip(),
                'evaluation_status': status,
                'evaluation_message': str(row['evaluation_message']) if pd.notna(row['evaluation_message']) else "",
                'market': str(row['market']).strip()
            })
        
        logger.info(f"Loaded {len(base_photos)} base photos for market '{market}'")
        return base_photos, ""
        
    except Exception as e:
        logger.error(f"Error loading CSV: {type(e).__name__}: {e}", exc_info=True)
        return [], f"Error loading CSV file: {str(e)}"

def infer_label(filename: str) -> str:
    """Infer label from filename (PASS or FAIL). Wrapped in try-except to prevent crashes."""
    try:
        name = os.path.basename(filename).upper()
        if name.startswith("PASS"):
            return "PASS"
        if name.startswith("FAIL"):
            return "FAIL"
        raise ValueError(f"Filename must start with PASS or FAIL: {os.path.basename(filename)}")
    except Exception as e:
        logger.warning(f"Error inferring label from filename '{filename}': {e}")
        return "FAIL"

def to_data_url(filepath_or_url: str, max_size_mb: float = 4.5) -> str:
    """Convert image file or URL to data URL, compressing if necessary."""
    is_url = filepath_or_url.startswith("http://") or filepath_or_url.startswith("https://")
    
    if is_url:
        download_start = time.time()
        logger.debug(f"Downloading image from URL: {filepath_or_url[:80]}...")
        try:
            response = requests.get(filepath_or_url, timeout=30)
            response.raise_for_status()
            original_data = response.content
            download_elapsed = time.time() - download_start
            logger.debug(f"Successfully downloaded {len(original_data)} bytes from URL in {download_elapsed:.2f}s")
            if download_elapsed > 2.0:
                logger.warning(f"⚠️ Slow URL download: {filepath_or_url[:80]}... took {download_elapsed:.2f}s")
            content_type = response.headers.get('content-type', '')
            if content_type and content_type.startswith('image/'):
                mime = content_type
            else:
                mime, _ = mimetypes.guess_type(filepath_or_url)
                if not mime or not mime.startswith("image/"):
                    mime = "image/jpeg"
        except Exception as e:
            logger.error(f"Error downloading image from URL {filepath_or_url}: {e}")
            raise RuntimeError(f"Failed to download image from URL: {e}")
    else:
        mime, _ = mimetypes.guess_type(filepath_or_url)
        if not mime or not mime.startswith("image/"):
            mime = "image/jpeg"
        with open(filepath_or_url, "rb") as f:
            original_data = f.read()
    
    display_name = os.path.basename(filepath_or_url) if not is_url else filepath_or_url.split("/")[-1]
    max_size_bytes = int(max_size_mb * 1024 * 1024)
    estimated_base64_size = len(original_data) * 4 // 3
    
    if estimated_base64_size <= max_size_bytes:
        b64 = base64.b64encode(original_data).decode("utf-8")
        file_size_kb = len(original_data) / 1024
        logger.debug(f"Image {display_name}: {file_size_kb:.1f}KB, no compression needed (fast path)")
    else:
        logger.info(f"Compressing image {display_name}: {len(original_data)} bytes -> target < {max_size_bytes} bytes")
        try:
            img = Image.open(BytesIO(original_data))
            
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")
            
            file_size_mb = len(original_data) / (1024 * 1024)
            
            if file_size_mb > 5:
                max_dimension = 1024
            elif file_size_mb > 2:
                max_dimension = 1536
            else:
                max_dimension = 2048
            
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                logger.debug(f"Resized image {display_name} from {img.size} to {new_size} (file was {file_size_mb:.2f}MB)")
            
            if file_size_mb > 5:
                initial_quality = 60
            elif file_size_mb > 2:
                initial_quality = 70
            else:
                initial_quality = 80
            quality = initial_quality
            
            for attempt in range(2):
                output = BytesIO()
                img.save(output, format="JPEG", quality=quality, optimize=False)
                compressed_data = output.getvalue()
                estimated_base64_size = len(compressed_data) * 4 // 3
                
                if estimated_base64_size <= max_size_bytes:
                    b64 = base64.b64encode(compressed_data).decode("utf-8")
                    logger.info(f"Compressed image {display_name}: {len(original_data)/1024:.1f}KB -> {len(compressed_data)/1024:.1f}KB (quality={quality})")
                    break
                else:
                    quality = max(40, quality - 25)
                    if attempt == 1:
                        b64 = base64.b64encode(compressed_data).decode("utf-8")
                        logger.warning(f"Image {display_name} still large after compression: {len(compressed_data)/1024:.1f}KB (quality={quality})")
        except Exception as e:
            logger.warning(f"Error compressing image {display_name}: {e}, using original")
            b64 = base64.b64encode(original_data).decode("utf-8")
    
    return f"data:{mime};base64,{b64}"

def parse_json_strict(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        return json.loads(text[s:e+1])
    raise ValueError("Model response is not valid JSON")

def compute_metrics(rows: List[Dict[str, Any]], ux_focus_pct: float = 0.0) -> Dict[str, Any]:
    tp = sum(1 for r in rows if r["gt"] == "PASS" and r["pred"] == "PASS")
    fn = sum(1 for r in rows if r["gt"] == "PASS" and r["pred"] == "FAIL")
    fp = sum(1 for r in rows if r["gt"] == "FAIL" and r["pred"] == "PASS")
    tn = sum(1 for r in rows if r["gt"] == "FAIL" and r["pred"] == "FAIL")

    recall = (tp / (tp + fn) * 100.0) if (tp + fn) else 0.0
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) else 0.0
    
    w_ux = ux_focus_pct / 100.0
    w_gp = 1.0 - w_ux
    weighted_score = (w_gp * recall + w_ux * precision)
    
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall_pass_pct": round(recall, 2),
        "precision_pass_pct": round(precision, 2),
        "weighted_score": round(weighted_score, 2),
        "ux_focus_pct": round(ux_focus_pct, 1),
    }

def openrouter_classify(system_prompt: str, user_prompt: str, image_data_url: str) -> Dict[str, Any]:
    logger.debug("Starting openrouter_classify")
    if not OPENROUTER_API_KEY:
        error_msg = "Missing OPENROUTER_API_KEY environment variable"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Streamlit Prompt Optimizer",
    }

    json_contract = """
Return ONLY valid JSON (no markdown, no extra text).
Schema:
{
  "decision": "PASS" | "FAIL",
  "reason": "short reason (max 25 words)"
}
""".strip()

    payload = {
        "model": MODEL_ID,
        "temperature": MODEL_TEMPERATURE,
        "messages": [
            {"role": "system", "content": (system_prompt or "").strip()},
            {"role": "user", "content": [
                {"type": "text", "text": (user_prompt or "").strip() + "\n\n" + json_contract},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]},
        ],
    }

    try:
        api_call_start = time.time()
        logger.debug(f"Calling OpenRouter API with model {MODEL_ID}")
        data = post_with_retries(url, headers, payload)
        api_call_elapsed = time.time() - api_call_start
        logger.debug(f"Received response from OpenRouter in {api_call_elapsed:.2f}s")
        if api_call_elapsed > 10.0:
            logger.warning(f"⚠️ Slow API call: took {api_call_elapsed:.2f}s")
        
        if "error" in data:
            error_info = data.get("error", {})
            error_msg = error_info.get("message", "Unknown error")
            if "metadata" in error_info and "raw" in error_info["metadata"]:
                try:
                    raw_error = json.loads(error_info["metadata"]["raw"])
                    if "error" in raw_error and "message" in raw_error["error"]:
                        error_msg = raw_error["error"]["message"]
                except:
                    pass
            raise RuntimeError(f"API returned error: {error_msg}")
        
        if "choices" not in data or len(data["choices"]) == 0:
            error_msg = f"Invalid response structure: {data}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        content = data["choices"][0]["message"]["content"]
        logger.debug(f"Response content length: {len(content)}")
        
        parsed = parse_json_strict(content)
        logger.debug(f"Parsed JSON successfully: {parsed}")

        decision = str(parsed.get("decision", "")).upper().strip()
        if decision not in ("PASS", "FAIL"):
            error_msg = f"Invalid decision: {decision}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        result = {"decision": decision, "reason": parsed.get("reason", "")}
        logger.debug(f"Classification result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in openrouter_classify: {type(e).__name__}: {str(e)}")
        raise

def analyze_error_patterns(errors: List[Dict[str, Any]], error_type: str = "errors") -> str:
    """Analyze error patterns from the reasons provided by the model."""
    if not errors:
        return f"No {error_type} to analyze."
    
    reasons = [e.get('reason', '').lower() for e in errors if e.get('reason')]
    if not reasons:
        return f"Found {len(errors)} {error_type}, but no reasons available for pattern analysis."
    
    keywords = {
        'crosswalk': ['crosswalk', 'crossing', 'pedestrian crossing'],
        'sidewalk': ['sidewalk', 'walkway', 'pedestrian path'],
        'blocking': ['blocking', 'blocked', 'obstruct', 'obstruction'],
        'driveway': ['driveway', 'drive way'],
        'entrance': ['entrance', 'entry', 'doorway'],
        'designated': ['designated', 'parking space', 'parking zone', 'marked'],
        'furniture zone': ['furniture zone', 'furniture'],
        'upright': ['upright', 'tipped', 'fallen', 'fallen over'],
        'visible': ['visible', 'cut off', 'partial', 'incomplete'],
        'ramp': ['ramp', 'handicap', 'accessible', 'ada'],
    }
    
    pattern_counts = {}
    for keyword, variations in keywords.items():
        count = sum(1 for reason in reasons if any(var in reason for var in variations))
        if count > 0:
            pattern_counts[keyword] = count
    
    if not pattern_counts:
        return f"Found {len(errors)} {error_type}, but no clear patterns detected in reasons."
    
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    top_patterns = sorted_patterns[:5]
    pattern_summary = "; ".join([f"{keyword} ({count} cases)" for keyword, count in top_patterns])
    return f"Common patterns: {pattern_summary}"

@dataclass
class ImageItem:
    file: str
    gt: str
    data_url: str = ""
    evaluation_message: str = ""
    
    def get_data_url(self) -> str:
        """Lazy load data URL - compute only when needed"""
        if not self.data_url:
            if not self.file.startswith(('http://', 'https://')):
                if not os.path.exists(self.file):
                    raise FileNotFoundError(
                        f"File no longer exists: {self.file}\n"
                        f"This should not happen with persistent uploads. File may have been manually deleted."
                    )
            self.data_url = to_data_url(self.file)
        return self.data_url

def evaluate_single_item(
    item: ImageItem,
    system_prompt: str,
    user_prompt: str,
    idx: int,
    total: int
) -> Dict[str, Any]:
    """Evaluate a single image item (used for parallel processing)"""
    if stop_event.is_set():
        raise InterruptedError("Optimization stopped by user")
    
    try:
        logger.debug(f"Evaluating item {idx}/{total}: {os.path.basename(item.file)}")
        out = openrouter_classify(system_prompt, user_prompt, item.get_data_url())
        pred = out["decision"]
        reason = out.get("reason", "")
        logger.debug(f"Item {idx} result: {pred}")
        return {
            "file": os.path.basename(item.file),
            "gt": item.gt,
            "pred": pred,
            "reason": reason,
            "error": False
        }
    except InterruptedError:
        raise
    except Exception as e:
        logger.warning(f"Error evaluating item {idx} ({os.path.basename(item.file)}): {type(e).__name__}: {str(e)}")
        return {
            "file": os.path.basename(item.file),
            "gt": item.gt,
            "pred": "FAIL",
            "reason": f"ERROR: {str(e)[:180]}",
            "error": True
        }

def evaluate_prompt(
    system_prompt: str,
    user_prompt: str,
    items: List[ImageItem],
    progress_bar=None,  # Streamlit progress bar
    status_text=None,   # Streamlit status text
    prefix: str = "Eval",
    ux_focus_pct: float = 0.0
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate prompt on items - adapted for Streamlit"""
    logger.info(f"Evaluating prompt on {len(items)} items (prefix: {prefix}) using {MAX_WORKERS} workers")
    rows: List[Dict[str, Any]] = []
    errors = 0
    total = len(items)
    
    completed = 0
    rows_dict = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(evaluate_single_item, it, system_prompt, user_prompt, idx, total): idx
            for idx, it in enumerate(items, start=1)
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            completed += 1
            
            if stop_event.is_set():
                logger.warning(f"Stop requested during evaluation at item {completed}/{total}")
                for f in future_to_idx:
                    f.cancel()
                raise InterruptedError("Optimization stopped by user")
            
            # Update Streamlit progress
            if progress_bar is not None:
                progress_bar.progress(completed / max(total, 1))
            if status_text is not None:
                status_text.text(f"{prefix}: {completed}/{total}")
            
            try:
                result = future.result(timeout=OPENROUTER_TIMEOUT_SECONDS + 10)
                rows_dict[idx] = result
                if result.get("error", False):
                    errors += 1
            except InterruptedError:
                raise
            except TimeoutError as e:
                logger.error(f"Timeout processing item {idx}: {e}")
                rows_dict[idx] = {
                    "file": os.path.basename(items[idx-1].file),
                    "gt": items[idx-1].gt,
                    "pred": "FAIL",
                    "reason": f"ERROR: Timeout after {OPENROUTER_TIMEOUT_SECONDS}s",
                    "error": True
                }
                errors += 1
            except Exception as e:
                logger.error(f"Unexpected error processing item {idx}: {type(e).__name__}: {e}", exc_info=True)
                rows_dict[idx] = {
                    "file": os.path.basename(items[idx-1].file),
                    "gt": items[idx-1].gt,
                    "pred": "FAIL",
                    "reason": f"ERROR: {str(e)[:180]}",
                    "error": True
                }
                errors += 1
    
    rows = []
    for idx in range(1, total + 1):
        if idx in rows_dict:
            row = rows_dict[idx].copy()
            row.pop("error", None)
            rows.append(row)
        else:
            logger.warning(f"Missing result for item {idx}, creating fallback")
            rows.append({
                "file": os.path.basename(items[idx-1].file) if idx <= len(items) else "unknown",
                "gt": items[idx-1].gt if idx <= len(items) else "FAIL",
                "pred": "FAIL",
                "reason": "ERROR: Missing result"
            })
            errors += 1
    
    if progress_bar is not None:
        progress_bar.progress(1.0)
    if status_text is not None:
        status_text.text(f"{prefix}: done")
    
    metrics = compute_metrics(rows, ux_focus_pct=ux_focus_pct)
    logger.info(f"Evaluation complete: {metrics} (errors: {errors})")
    metrics["invalid_json_or_error_count"] = errors
    return rows, metrics

def better(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    ma = a["metrics"]
    mb = b["metrics"]
    score_a = ma.get("weighted_score", 0)
    score_b = mb.get("weighted_score", 0)
    return a if score_a >= score_b else b

def propose_next_user_prompt(
    system_prompt: str,
    current_user_prompt: str,
    false_negatives: List[Dict[str, Any]],
    false_positives: List[Dict[str, Any]],
    current_metrics: Dict[str, Any],
    ux_focus_pct: float = 0.0,
) -> str:
    """Propose next user prompt using LLM optimizer"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Streamlit Prompt Optimizer",
    }

    fn_lines = "\n".join([
        f"- {x['file']}: predicted FAIL but should PASS. Model reason: {x.get('reason','')}"
        for x in false_negatives[:20]
    ]) or "- None"

    fp_lines = "\n".join([
        f"- {x['file']}: predicted PASS but should FAIL. Model reason: {x.get('reason','')}"
        for x in false_positives[:20]
    ]) or "- None"

    base_rules_text = ""
    try:
        if BASE_RULES:
            base_rules_text = "\n\nBASE RULES (must be preserved in all prompts - these are included at the beginning of the user prompt and must always be kept):\n" + "\n".join([f"- {rule}" for rule in BASE_RULES])
    except NameError:
        pass
    
    w_ux = ux_focus_pct / 100.0
    w_gp = 1.0 - w_ux
    focus_description = f"""
Optimization Focus: {ux_focus_pct}% UX / {100-ux_focus_pct}% GP
- GP Focus ({100-ux_focus_pct}%): Maximize recall (catch valid parking), minimize false negatives
- UX Focus ({ux_focus_pct}%): Maximize precision (avoid false alarms), minimize false positives
Weighted Score = {w_gp:.1%} × Recall + {w_ux:.1%} × Precision
Maximize this weighted score.
"""
    
    fn_patterns = analyze_error_patterns(false_negatives, "false negatives")
    fp_patterns = analyze_error_patterns(false_positives, "false positives")
    
    current_recall = current_metrics.get('recall_pass_pct', 0)
    current_precision = current_metrics.get('precision_pass_pct', 0)
    current_weighted = current_metrics.get('weighted_score', 0)
    
    focus_type = "false negatives (recall)" if w_gp > w_ux else "false positives (precision)" if w_ux > w_gp else "both (balanced)"
    
    try:
        target_recall = TARGET_RECALL
        target_precision = TARGET_PRECISION
    except NameError:
        target_recall = 95.0
        target_precision = 95.0
    
    recall_gap = max(0, target_recall - current_recall)
    precision_gap = max(0, target_precision - current_precision)
    
    if recall_gap > precision_gap:
        primary_focus = f"Recall needs {recall_gap:.1f}% improvement (currently {current_recall}%, target {target_recall}%)"
        secondary_focus = f"Precision needs {precision_gap:.1f}% improvement (currently {current_precision}%, target {target_precision}%)"
    else:
        primary_focus = f"Precision needs {precision_gap:.1f}% improvement (currently {current_precision}%, target {target_precision}%)"
        secondary_focus = f"Recall needs {recall_gap:.1f}% improvement (currently {current_recall}%, target {target_recall}%)"
    
    prompt = f"""
You are an expert prompt engineer specializing in image classification tasks.

PRIMARY GOAL: Achieve {target_recall}% recall AND {target_precision}% precision simultaneously.

Current Performance:
- Recall: {current_recall}% (Target: {target_recall}%, Gap: {recall_gap:.1f}%)
- Precision: {current_precision}% (Target: {target_precision}%, Gap: {precision_gap:.1f}%)
- Weighted Score: {current_weighted}%

Optimization Strategy:
{focus_description}

Priority Focus:
1. {primary_focus}
2. {secondary_focus}

You MUST output ONLY the revised USER PROMPT text (no commentary, no markdown).

CRITICAL REQUIREMENTS:
1. You MUST output ONLY the revised USER PROMPT text (no commentary, no markdown, no explanations).
2. The BASE RULES must be included at the BEGINNING of the user prompt and preserved EXACTLY as shown.
3. Do not change the system prompt structure.
4. Make specific, actionable improvements based on the error patterns below.
5. Be precise and clear - avoid vague language.
6. If false negatives are the issue, make the prompt MORE lenient for valid parking scenarios.
7. If false positives are the issue, make the prompt MORE strict about what constitutes valid parking.
8. Balance both concerns - don't optimize for one metric at the expense of the other.

BASE RULES (MUST BE PRESERVED):
{base_rules_text}

Current USER PROMPT:
{current_user_prompt}

ERROR ANALYSIS - False Negatives (GT PASS, predicted FAIL) - {len(false_negatives)} total:
These are valid parking cases that were incorrectly rejected. The model is being too strict.

Pattern Analysis:
{fn_patterns}

Specific Examples ({min(len(false_negatives), 20)} of {len(false_negatives)}):
{fn_lines}

ERROR ANALYSIS - False Positives (GT FAIL, predicted PASS) - {len(false_positives)} total:
These are invalid parking cases that were incorrectly accepted. The model is being too lenient.

Pattern Analysis:
{fp_patterns}

Specific Examples ({min(len(false_positives), 20)} of {len(false_positives)}):
{fp_lines}

INSTRUCTIONS:
1. Analyze the error patterns above carefully.
2. Identify the root causes of both false negatives and false positives.
3. Revise the USER PROMPT to address BOTH issues simultaneously.
4. Make the prompt more specific about edge cases.
5. Add clarifications that will help the model distinguish between similar-looking valid and invalid scenarios.
6. Ensure the prompt guides the model to achieve {target_recall}% recall AND {target_precision}% precision.

Output ONLY the complete revised USER PROMPT (including BASE RULES at the beginning).
""".strip()

    payload = {
        "model": MODEL_ID,
        "temperature": MODEL_TEMPERATURE,
        "messages": [
            {"role": "system", "content": (system_prompt or "").strip()},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        logger.debug("Calling optimizer LLM")
        data = post_with_retries(url, headers, payload)
        
        if "choices" not in data or len(data["choices"]) == 0:
            error_msg = f"Invalid response structure from optimizer: {data}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        new_prompt = (data["choices"][0]["message"]["content"] or "").strip()
        if not new_prompt:
            error_msg = "Empty prompt returned from optimizer step"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info(f"New prompt generated (length: {len(new_prompt)} chars)")
        return new_prompt
    except Exception as e:
        logger.error(f"Error in propose_next_user_prompt: {type(e).__name__}: {str(e)}")
        raise

def cleanup_persistent_uploads():
    """Clean up the persistent uploads directory."""
    persistent_uploads_dir = "./persistent_uploads"
    try:
        if os.path.exists(persistent_uploads_dir):
            try:
                for filename in os.listdir(persistent_uploads_dir):
                    file_path = os.path.join(persistent_uploads_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.debug(f"Could not remove file {filename}: {e}")
                os.rmdir(persistent_uploads_dir)
            except Exception:
                shutil.rmtree(persistent_uploads_dir)
            logger.info(f"Cleaned up persistent uploads directory: {persistent_uploads_dir}")
    except Exception as e:
        logger.warning(f"Error cleaning up persistent uploads directory: {e}", exc_info=True)

# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(
    page_title="Prompt Optimizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'optimization_running' not in st.session_state:
    st.session_state.optimization_running = False
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'market_preview' not in st.session_state:
    st.session_state.market_preview = ""

st.title("🚀 Prompt Optimizer — Weighted Recall/Precision Optimization")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    ux_focus_pct = st.slider(
        "UX Focus (%)",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        help="0% = GP focused (maximize recall, minimize FN) | 100% = UX focused (maximize precision, minimize FP)"
    )
    
    st.markdown("---")
    st.markdown("### Status")
    status_placeholder = st.empty()
    progress_placeholder = st.empty()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input")
    
    # Market input
    market_col1, market_col2 = st.columns([4, 1])
    with market_col1:
        market_input = st.text_input(
            "Market Name (optional)",
            placeholder="Enter market name (e.g., Denver, Rome) to load base photos from CSV",
            help="Case-insensitive. Loads photos from 'ai photo review - examples.csv'"
        )
    with market_col2:
        load_market_btn = st.button("Load Photos", type="secondary")
    
    # Market preview
    if load_market_btn and market_input:
        base_photos, error_msg = load_base_photos_from_csv(market_input.strip())
        if error_msg:
            st.session_state.market_preview = f"❌ {error_msg}"
        elif not base_photos:
            st.session_state.market_preview = f"⚠️ No photos found for market '{market_input}'."
        else:
            pass_count = sum(1 for p in base_photos if p['evaluation_status'] == 'PASS')
            fail_count = sum(1 for p in base_photos if p['evaluation_status'] == 'FAIL')
            st.session_state.market_preview = f"✅ Found {len(base_photos)} photos for market '{market_input}': {pass_count} PASS, {fail_count} FAIL"
    
    st.text_area("Market Preview", value=st.session_state.market_preview, height=60, disabled=True)
    
    st.markdown("---")
    
    # File upload - Streamlit handles this much better than Gradio!
    st.markdown("**Upload Images (optional)**")
    st.info("📁 Filenames must start with PASS_ or FAIL_")
    
    uploaded_files = st.file_uploader(
        "Upload images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload images with filenames starting with PASS_ or FAIL_"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        with st.expander("View uploaded files"):
            for file in uploaded_files:
                st.text(f"• {file.name} ({file.size / 1024 / 1024:.2f} MB)")

with col2:
    st.subheader("Information")
    market_display = st.text_input("Market", value="None", disabled=True)
    photo_stats = st.text_area("Photo Statistics", value="", height=60, disabled=True)

# Action buttons
col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    run_btn = st.button("🚀 Run Optimization", type="primary", use_container_width=True)
with col_btn2:
    stop_btn = st.button("⏹️ Stop", use_container_width=True, disabled=not st.session_state.optimization_running)

# Handle stop button
if stop_btn and st.session_state.optimization_running:
    stop_event.set()
    st.session_state.stop_requested = True
    st.session_state.optimization_running = False
    status_placeholder.warning("⚠️ Stop requested. Optimization will stop at next checkpoint...")
    st.rerun()

# Handle run button
if run_btn:
    # Check if we have input
    if not market_input and not uploaded_files:
        st.error("❌ Please upload images or enter a market name")
        st.stop()
    
    # Set running state
    st.session_state.optimization_running = True
    st.session_state.stop_requested = False
    stop_event.clear()
    
    # Clean up persistent uploads at start
    cleanup_persistent_uploads()
    
    # Save uploaded files to temp directory (Streamlit file_uploader returns file objects)
    temp_files = []
    if uploaded_files:
        temp_dir = tempfile.mkdtemp(prefix="streamlit_uploads_")
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            temp_files.append(temp_path)
    
    # Process files in parallel (like in notebook)
    processed_files = None
    if temp_files:
        persistent_uploads_dir = "./persistent_uploads"
        os.makedirs(persistent_uploads_dir, exist_ok=True)
        processed_files = []
        
        def process_single_file(args):
            idx, temp_path = args
            try:
                if not temp_path or not os.path.exists(temp_path):
                    return None
                
                original_name = os.path.basename(temp_path)
                persistent_path = os.path.join(persistent_uploads_dir, original_name)
                
                counter = 1
                base_name, ext = os.path.splitext(original_name)
                while os.path.exists(persistent_path):
                    persistent_path = os.path.join(persistent_uploads_dir, f"{base_name}_{counter}{ext}")
                    counter += 1
                
                shutil.copy2(temp_path, persistent_path)
                
                # Convert to data URL immediately
                data_url = to_data_url(persistent_path)
                
                return {
                    "file": persistent_path,
                    "data_url": data_url,
                    "original_name": original_name
                }
            except Exception as e:
                logger.error(f"Error processing file {temp_path}: {e}")
                return None
        
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(temp_files))) as executor:
            results = list(executor.map(process_single_file, enumerate(temp_files, 1)))
        
        for result in results:
            if result:
                processed_files.append(result)
    
    # Create progress tracking for Streamlit
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    # Run optimizer (adapted for Streamlit)
    try:
        status_text.text("🔄 Starting optimization...")
        
        # Build items list
        items: List[ImageItem] = []
        pass_n = 0
        fail_n = 0
        bad = 0
        base_photos_count = 0
        market_name = ""
        
        # Process uploaded files first
        if processed_files:
            for processed_file in processed_files:
                try:
                    file_path = processed_file["file"]
                    data_url = processed_file.get("data_url", "")
                    gt = infer_label(file_path)
                    
                    if gt == "PASS":
                        pass_n += 1
                    else:
                        fail_n += 1
                    
                    items.append(ImageItem(
                        file=file_path,
                        gt=gt,
                        data_url=data_url
                    ))
                except Exception as e:
                    bad += 1
                    logger.warning(f"Error adding processed file: {e}")
        
        uploaded_count = len(processed_files) if processed_files else 0
        
        # Load base photos from CSV if market provided
        if market_input and market_input.strip():
            market_name = market_input.strip()
            status_text.text("Loading base photos from CSV...")
            base_photos, csv_error = load_base_photos_from_csv(market_name)
            
            if csv_error:
                st.error(f"Error loading base photos: {csv_error}")
                st.stop()
            
            if base_photos:
                base_photos_count = len(base_photos)
                status_text.text(f"Pre-downloading {base_photos_count} base photos from URLs...")
                
                def process_base_photo(photo):
                    try:
                        gt = photo['evaluation_status']
                        data_url = to_data_url(photo['image_url'])
                        return {
                            "file": photo['image_url'],
                            "gt": gt,
                            "data_url": data_url,
                            "evaluation_message": photo['evaluation_message'],
                            "success": True
                        }
                    except Exception as e:
                        logger.warning(f"Error processing base photo: {e}")
                        return {"success": False}
                
                with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, base_photos_count)) as executor:
                    base_results = list(executor.map(process_base_photo, base_photos))
                
                for result in base_results:
                    if result["success"]:
                        if result["gt"] == "PASS":
                            pass_n += 1
                        else:
                            fail_n += 1
                        items.append(ImageItem(
                            file=result["file"],
                            gt=result["gt"],
                            data_url=result["data_url"],
                            evaluation_message=result["evaluation_message"]
                        ))
                    else:
                        bad += 1
        
        if not items:
            st.error("No valid images found. Upload images or enter a market name.")
            st.stop()
        
        if pass_n < MIN_PASS_IMAGES_REQUIRED:
            st.error(f"Not enough PASS images (found {pass_n}, need at least {MIN_PASS_IMAGES_REQUIRED})")
            st.stop()
        
        # Use default prompts
        system_prompt = DEFAULT_SYSTEM_PROMPT
        base_user_prompt = DEFAULT_BASE_USER_PROMPT
        
        history: List[Dict[str, Any]] = []
        
        # Iteration 0
        status_text.text("Iteration 0: evaluating base prompt...")
        rows0, m0 = evaluate_prompt(system_prompt, base_user_prompt, items, 
                                    progress_bar=progress_bar, status_text=status_text, 
                                    prefix="Iter 0", ux_focus_pct=ux_focus_pct)
        history.append({"iteration": 0, "user_prompt": base_user_prompt, "metrics": m0, "rows": rows0})
        best = history[0]
        current_prompt = base_user_prompt
        
        # Optimization iterations
        for i in range(1, OPTIMIZATION_ITERATIONS + 1):
            if stop_event.is_set():
                break
            
            status_text.text(f"Iteration {i}: building edge cases...")
            last_rows = history[-1]["rows"]
            false_neg = [r for r in last_rows if r["gt"] == "PASS" and r["pred"] == "FAIL"]
            false_pos = [r for r in last_rows if r["gt"] == "FAIL" and r["pred"] == "PASS"]
            
            if stop_event.is_set():
                break
            
            status_text.text(f"Iteration {i}: proposing new prompt...")
            current_metrics = history[-1]["metrics"]
            new_prompt = propose_next_user_prompt(
                system_prompt=system_prompt,
                current_user_prompt=current_prompt,
                false_negatives=false_neg,
                false_positives=false_pos,
                current_metrics=current_metrics,
                ux_focus_pct=ux_focus_pct,
            )
            
            if stop_event.is_set():
                break
            
            status_text.text(f"Iteration {i}: evaluating new prompt...")
            rows_i, m_i = evaluate_prompt(system_prompt, new_prompt, items,
                                         progress_bar=progress_bar, status_text=status_text,
                                         prefix=f"Iter {i}", ux_focus_pct=ux_focus_pct)
            candidate = {"iteration": i, "user_prompt": new_prompt, "metrics": m_i, "rows": rows_i}
            history.append(candidate)
            
            recall_i = m_i.get('recall_pass_pct', 0)
            precision_i = m_i.get('precision_pass_pct', 0)
            if recall_i >= EARLY_STOP_THRESHOLD and precision_i >= EARLY_STOP_THRESHOLD:
                logger.info(f"🎯 TARGET ACHIEVED! Stopping early")
                break
            
            best = better(best, candidate)
            current_prompt = new_prompt
        
        # Build results
        best_metrics = best["metrics"]
        best_prompt_text = best["user_prompt"]
        
        # Create dataframes
        history_data = []
        for h in history:
            m = h["metrics"]
            history_data.append({
                "Iteration": h["iteration"],
                "Recall %": m.get("recall_pass_pct", 0),
                "Precision %": m.get("precision_pass_pct", 0),
                "Weighted Score %": m.get("weighted_score", 0),
                "TP": m.get("tp", 0),
                "FN": m.get("fn", 0),
                "FP": m.get("fp", 0),
                "TN": m.get("tn", 0),
            })
        df_history = pd.DataFrame(history_data)
        
        fn_rows = [r for r in best["rows"] if r["gt"] == "PASS" and r["pred"] == "FAIL"]
        fp_rows = [r for r in best["rows"] if r["gt"] == "FAIL" and r["pred"] == "PASS"]
        
        df_fn = pd.DataFrame(fn_rows) if fn_rows else pd.DataFrame()
        df_fp = pd.DataFrame(fp_rows) if fp_rows else pd.DataFrame()
        
        # Build summary
        market_info = f"Market: {market_name}\n" if market_name else ""
        base_photos_info = f"Base photos: {base_photos_count}\n" if base_photos_count > 0 else ""
        uploaded_info = f"Uploaded photos: {uploaded_count}\n" if uploaded_count > 0 else ""
        
        summary = (
            f"Model: {MODEL_ID} | Temp: {MODEL_TEMPERATURE}\n"
            f"{market_info}{base_photos_info}{uploaded_info}"
            f"Dataset: total={len(items)} PASS={pass_n} FAIL={fail_n} bad={bad}\n"
            f"Optimization Focus: {best_metrics.get('ux_focus_pct', 0)}% UX / {100-best_metrics.get('ux_focus_pct', 0)}% GP\n"
            f"Best iteration: {best['iteration']}\n"
            f"Weighted Score: {best_metrics.get('weighted_score', 0)}%\n"
            f"Recall(PASS): {best_metrics['recall_pass_pct']}%\n"
            f"Precision(PASS): {best_metrics['precision_pass_pct']}%\n"
            f"TP:{best_metrics['tp']} FN:{best_metrics['fn']} FP:{best_metrics['fp']} TN:{best_metrics['tn']}\n"
            f"Invalid/Errors: {best_metrics.get('invalid_json_or_error_count', 0)}\n"
        )
        
        # Store results in session state
        st.session_state.results = {
            'summary': summary,
            'best_prompt': best_prompt_text,
            'history_df': df_history,
            'fn_df': df_fn,
            'fp_df': df_fp,
            'market_display': market_name if market_name else "None",
            'photo_stats': f"Base photos: {base_photos_count} | Uploaded: {uploaded_count} | Total: {len(items)} | PASS: {pass_n} | FAIL: {fail_n}"
        }
        
        st.session_state.optimization_running = False
        status_text.text("✅ Optimization completed!")
        progress_bar.progress(1.0)
        
        # Clean up
        cleanup_persistent_uploads()
        if temp_files:
            shutil.rmtree(os.path.dirname(temp_files[0]), ignore_errors=True)
        
        st.rerun()
        
    except InterruptedError:
        st.session_state.optimization_running = False
        status_text.warning("⚠️ Optimization stopped by user")
        cleanup_persistent_uploads()
        st.rerun()
    except Exception as e:
        st.session_state.optimization_running = False
        status_text.error(f"❌ Error: {type(e).__name__}: {str(e)}")
        logger.error(f"Error in optimization: {e}", exc_info=True)
        cleanup_persistent_uploads()
        st.rerun()

# Display results
if st.session_state.results:
    st.markdown("---")
    st.subheader("Results")
    
    results = st.session_state.results
    
    st.text_area("Summary", value=results.get('summary', ''), height=150)
    st.text_area("Best Optimized Prompt", value=results.get('best_prompt', ''), height=200)
    
    col_df1, col_df2 = st.columns(2)
    
    with col_df1:
        if 'history_df' in results and not results['history_df'].empty:
            st.dataframe(results['history_df'], use_container_width=True)
    
    with col_df2:
        if 'fn_df' in results and not results['fn_df'].empty:
            st.dataframe(results['fn_df'], use_container_width=True)
    
    if 'fp_df' in results and not results['fp_df'].empty:
        st.dataframe(results['fp_df'], use_container_width=True)
    
    # Update info displays
    if results.get('market_display'):
        market_display = results['market_display']
    if results.get('photo_stats'):
        photo_stats = results['photo_stats']
