"""
Utility functions for AI photo evaluation app.
Extracted to simplify the main notebook.
"""
import os
import json
import base64
import hashlib
import logging
import tempfile
import zipfile
import time
import requests
import mimetypes
from typing import Dict, Any, List, Optional, Tuple
from io import BytesIO
from PIL import Image
import pandas as pd

logger = logging.getLogger(__name__)

# ==================== HTTP & API Helpers ====================

def post_with_retries(
    url: str,
    headers: dict,
    payload: dict,
    timeout_s: int = 120,
    retries: int = 3,
) -> dict:
    """POST with retries + exponential backoff. Returns response.json() on success."""
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

    raise RuntimeError(f"Request failed after {retries} attempts. Last error: {last_err}")

# ==================== CSV Loading ====================

def load_base_photos_from_csv(market: str, csv_path: str = "ai photo review - examples.csv") -> Tuple[List[Dict[str, Any]], str]:
    """Load base photos from CSV file filtered by market (case-insensitive)."""
    if not market or not market.strip():
        return [], ""
    
    market = market.strip()
    possible_paths = [csv_path, os.path.join("..", csv_path)]
    
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
        
        # Filter by market (case-insensitive)
        filtered = df[df['market'].str.strip().str.lower() == market.lower()]
        logger.info(f"Found {len(filtered)} photos for market '{market}'")
        
        photos = []
        for _, row in filtered.iterrows():
            photos.append({
                'image_url': row['image_url'],
                'evaluation_status': row['evaluation_status'],
                'evaluation_message': row.get('evaluation_message', ''),
                'market': row['market']
            })
        
        return photos, ""
    except Exception as e:
        return [], f"Error reading CSV file: {str(e)}"

# ==================== Image Processing ====================

def infer_label(filename: str) -> str:
    """Infer PASS/FAIL label from filename (must start with PASS_ or FAIL_)."""
    basename = os.path.basename(filename).upper()
    if basename.startswith("PASS_"):
        return "PASS"
    elif basename.startswith("FAIL_"):
        return "FAIL"
    return ""

def to_data_url(filepath_or_url: str, max_size_mb: Optional[float] = None) -> str:
    """Convert image file or URL to base64 data URL with compression."""
    if max_size_mb is None:
        max_size_mb = 3.5
    
    max_size_bytes = int(max_size_mb * 1024 * 1024)
    
    # Handle URLs
    if filepath_or_url.startswith('http://') or filepath_or_url.startswith('https://'):
        try:
            response = requests.get(filepath_or_url, timeout=30)
            response.raise_for_status()
            original_data = response.content
            display_name = filepath_or_url[:50]
            
            # Determine MIME type
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
        # Local file path
        with open(filepath_or_url, 'rb') as f:
            original_data = f.read()
        display_name = os.path.basename(filepath_or_url)
        mime, _ = mimetypes.guess_type(filepath_or_url)
        if not mime or not mime.startswith("image/"):
            mime = "image/jpeg"
    
    # Fast path: if already small enough, return immediately
    if len(original_data) <= max_size_bytes:
        b64 = base64.b64encode(original_data).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    
    # Need to compress
    try:
        img = Image.open(BytesIO(original_data))
        original_format = img.format or "JPEG"
        
        # Determine compression settings based on file size
        file_size_mb = len(original_data) / (1024 * 1024)
        if file_size_mb > 5:
            max_dimension = 1200
            initial_quality = 50
        elif file_size_mb > 2:
            max_dimension = 1600
            initial_quality = 60
        else:
            max_dimension = 1600
            initial_quality = 70
        
        # Resize if needed
        if max(img.size) > max_dimension:
            ratio = max_dimension / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        quality = initial_quality
        for attempt in range(2):
            output = BytesIO()
            img.save(output, format=original_format, quality=quality, optimize=True)
            compressed_data = output.getvalue()
            
            if len(compressed_data) <= max_size_bytes:
                b64 = base64.b64encode(compressed_data).decode("utf-8")
                logger.info(f"Compressed image {display_name}: {len(original_data)/1024:.1f}KB -> {len(compressed_data)/1024:.1f}KB (quality={quality})")
                return f"data:{mime};base64,{b64}"
            else:
                quality = max(35, quality - 30)
                if attempt == 1:
                    b64 = base64.b64encode(compressed_data).decode("utf-8")
                    logger.warning(f"Image {display_name} still large after compression: {len(compressed_data)/1024:.1f}KB (quality={quality})")
                    return f"data:{mime};base64,{b64}"
        
        # Fallback
        b64 = base64.b64encode(original_data).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        logger.error(f"Error compressing image {display_name}: {e}")
        # Fallback to original
        b64 = base64.b64encode(original_data).decode("utf-8")
        return f"data:{mime};base64,{b64}"

# ==================== JSON & Parsing ====================

def parse_json_strict(text: str) -> Dict[str, Any]:
    """Parse JSON with strict error handling."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)}")

# ==================== File Operations ====================

def cleanup_persistent_uploads():
    """Clean up persistent uploads directory."""
    persistent_uploads_dir = "./persistent_uploads"
    try:
        if os.path.exists(persistent_uploads_dir):
            import shutil
            shutil.rmtree(persistent_uploads_dir)
            logger.info(f"Cleaned up persistent uploads directory: {persistent_uploads_dir}")
        else:
            logger.debug(f"Persistent uploads directory does not exist: {persistent_uploads_dir}")
    except Exception as e:
        logger.warning(f"Error cleaning up persistent uploads directory: {e}")

def create_error_images_zip(rows_with_paths: List[Dict[str, Any]], error_type: str) -> Optional[str]:
    """Create a zip file containing images from error rows (FN or FP)."""
    if not rows_with_paths or len(rows_with_paths) == 0:
        return None
    
    try:
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time())
        zip_filename = f"{error_type}_images_{timestamp}.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            added_count = 0
            for row in rows_with_paths:
                file_path = row.get('file_path') or row.get('file')
                if not file_path:
                    continue
                
                # Handle URLs
                if file_path.startswith('http://') or file_path.startswith('https://'):
                    try:
                        response = requests.get(file_path, timeout=30)
                        response.raise_for_status()
                        img_data = response.content
                        
                        filename_from_url = os.path.basename(file_path.split('?')[0])
                        if not filename_from_url or '.' not in filename_from_url:
                            filename_from_url = f"image_{added_count}.jpg"
                        
                        if 'gt' in row and 'pred' in row:
                            name, ext = os.path.splitext(filename_from_url)
                            filename_in_zip = f"{name}_GT{row['gt']}_PRED{row['pred']}{ext}"
                        else:
                            filename_in_zip = filename_from_url
                        
                        zipf.writestr(filename_in_zip, img_data)
                        added_count += 1
                    except Exception as e:
                        logger.warning(f"Could not download image from URL {file_path}: {e}")
                        continue
                else:
                    # Handle local files
                    if not os.path.isabs(file_path):
                        possible_paths = [
                            file_path,
                            os.path.join('./persistent_uploads', file_path),
                            os.path.join('.', file_path),
                        ]
                        found = False
                        for possible in possible_paths:
                            if os.path.exists(possible) and os.path.isfile(possible):
                                file_path = possible
                                found = True
                                break
                        if not found:
                            logger.warning(f"Could not find file: {row.get('file', 'unknown')}")
                            continue
                    
                    if os.path.exists(file_path) and os.path.isfile(file_path):
                        filename_in_zip = os.path.basename(file_path)
                        if 'gt' in row and 'pred' in row:
                            name, ext = os.path.splitext(filename_in_zip)
                            filename_in_zip = f"{name}_GT{row['gt']}_PRED{row['pred']}{ext}"
                        
                        zipf.write(file_path, filename_in_zip)
                        added_count += 1
                    else:
                        logger.warning(f"File not found: {file_path}")
        
        if added_count == 0:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return None
        
        logger.info(f"Created {error_type.upper()} zip file: {zip_path} with {added_count} images")
        return zip_path
    except Exception as e:
        logger.error(f"Error creating {error_type} zip file: {e}", exc_info=True)
        return None
