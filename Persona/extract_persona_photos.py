#!/usr/bin/env python3
"""
Persona API Photo Extractor
===========================
Extracts front photo URLs from Persona inquiries and saves them to a CSV file.

This script is designed to be easy to use and share with team members.
Simply update the file paths below and run the script.
"""

import os
import csv
import time
from pathlib import Path
from typing import Dict, Any, Optional

import requests
from dotenv import load_dotenv

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR TEAMMATE
# ============================================================================

# Folder where input and output CSV files will be stored
# This makes it easy to find files for Google Sheets sync
# Default: same folder as this script. Change to Desktop folder if preferred:
# DATA_FOLDER = Path.home() / "Desktop" / "Persona_Data"
DATA_FOLDER = Path(__file__).parent  # Use same folder as script

# Input CSV file name (must have 'inquiry_id' column)
# Common names: "Persona inquiries.csv" or "Persona_inquiries.csv"
INPUT_CSV_NAME = "Persona inquiries.csv"

# Output CSV file name (will be created automatically)
OUTPUT_CSV_NAME = "inquiry_front_photos.csv"

# ============================================================================
# API SETTINGS
# ============================================================================

REQUEST_TIMEOUT = 30
SLEEP_S = 0.25  # Delay between API calls (be kind to API)

# ============================================================================
# PERSONA API FUNCTIONS
# ============================================================================

def persona_get_inquiry(inquiry_id: str, include_verifications: bool = True) -> Dict[str, Any]:
    """Retrieve an inquiry from Persona API."""
    from urllib.parse import quote
    encoded_id = quote(inquiry_id, safe='')
    
    url = f"https://api.withpersona.com/api/v1/inquiries/{encoded_id}"
    headers = {
        "Authorization": f"Bearer {PERSONA_API_KEY}",
        "Accept": "application/json",
    }
    
    params = {}
    if include_verifications:
        params["include"] = "verifications"
    
    r = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    
    if r.status_code == 404:
        raise ValueError(f"Inquiry {inquiry_id} not found (404).")
    
    r.raise_for_status()
    return r.json()


def get_last_government_id_verification(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get the last (most recent) verification/government-id from inquiry payload."""
    verifications = []
    included = payload.get("included", [])
    
    for item in included:
        if isinstance(item, dict):
            item_type = item.get("type", "")
            if item_type == "verification/government-id":
                verifications.append(item)
    
    return verifications[-1] if verifications else None


def extract_front_photo_url_from_last_verification(payload: Dict[str, Any]) -> Optional[str]:
    """Extract front-photo-url from the last (most recent) government-id verification."""
    verification = get_last_government_id_verification(payload)
    
    if verification:
        attributes = verification.get("attributes", {})
        front_photo_url = attributes.get("front-photo-url")
        if isinstance(front_photo_url, str) and front_photo_url:
            return front_photo_url
    
    return None


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def extract_front_photos_from_csv(input_csv: str, output_csv: str):
    """Read inquiry IDs from CSV and extract front-photo-url from the last verification."""
    fieldnames = ["file_name", "inquiry_id", "front_photo_url", "status"]
    
    # Read input CSV
    with open(input_csv, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        
        if not reader.fieldnames:
            raise ValueError(f"CSV file {input_csv} appears to be empty or invalid.")
        
        # Normalize column names for case-insensitive matching
        fieldnames_lower = {name.lower().strip(): name for name in reader.fieldnames}
        
        # Try to find inquiry_id and file_name columns
        inquiry_id_key = None
        file_name_key = None
        
        inquiry_id_variants = ["inquiry_id", "inquiry id", "inquiryid", "id", "inquiry"]
        file_name_variants = ["file name", "filename", "file_name", "File name"]
        
        for variant in inquiry_id_variants:
            variant_lower = variant.lower().strip()
            if variant_lower in fieldnames_lower:
                inquiry_id_key = fieldnames_lower[variant_lower]
                break
        
        for variant in file_name_variants:
            variant_lower = variant.lower().strip()
            if variant_lower in fieldnames_lower:
                file_name_key = fieldnames_lower[variant_lower]
                break
        
        if not inquiry_id_key:
            print(f"Available columns: {', '.join(reader.fieldnames)}")
            raise ValueError(f"Could not find inquiry_id column. Looking for: {', '.join(inquiry_id_variants)}")
        
        if file_name_key:
            print(f"Using columns: '{inquiry_id_key}' for inquiry_id, '{file_name_key}' for file_name\n")
        else:
            print(f"Using column: '{inquiry_id_key}' for inquiry_id (file_name column not found, will use inquiry_id as name)\n")
        
        # Extract inquiries
        inquiries = []
        for row in reader:
            inquiry_id = row.get(inquiry_id_key, "").strip()
            file_name = row.get(file_name_key, "").strip() if file_name_key else inquiry_id
            
            if inquiry_id:
                inquiries.append({"inquiry_id": inquiry_id, "file_name": file_name or inquiry_id})
    
    if not inquiries:
        raise ValueError(f"No inquiries found in {input_csv}. Check that 'inquiry_id' column exists and has data.")
    
    print(f"Found {len(inquiries)} inquiries to process\n")
    
    # Process inquiries and save results
    with open(output_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, inquiry in enumerate(inquiries, start=1):
            inquiry_id = inquiry["inquiry_id"]
            file_name = inquiry["file_name"]
            
            out = {
                "file_name": file_name,
                "inquiry_id": inquiry_id,
                "front_photo_url": "",
                "status": "OK"
            }
            
            print(f"[{i}/{len(inquiries)}] {file_name} ({inquiry_id})... ", end="", flush=True)
            try:
                payload = persona_get_inquiry(inquiry_id)
                front_photo_url = extract_front_photo_url_from_last_verification(payload)
                
                if front_photo_url:
                    out["front_photo_url"] = front_photo_url
                    print(f"✅ Found front photo")
                else:
                    out["status"] = "No government-id verification or no front photo"
                    print(f"⚠️ No front photo found")
                    
            except ValueError as e:
                out["status"] = f"Not found: {e}"
                print(f"❌ {e}")
            except requests.exceptions.HTTPError as e:
                status = getattr(e.response, 'status_code', 'unknown')
                out["status"] = f"HTTP {status}: {str(e)[:100]}"
                print(f"❌ HTTP {status}")
            except Exception as e:
                out["status"] = f"Error: {type(e).__name__}: {str(e)[:100]}"
                print(f"❌ {type(e).__name__}")
            
            writer.writerow(out)
            time.sleep(SLEEP_S)
    
    print(f"\n✅ Done! Front photo URLs saved to: {output_csv}")
    
    # Summary (using pandas if available, otherwise basic count)
    try:
        import pandas as pd
        summary_df = pd.read_csv(output_csv)
        total = len(summary_df)
        with_photos = len(summary_df[summary_df['front_photo_url'].notna() & (summary_df['front_photo_url'] != '')])
        without_photos = total - with_photos
        
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total inquiries processed: {total}")
        print(f"Inquiries with front photo: {with_photos} ({with_photos/total*100:.1f}%)")
        print(f"Inquiries without front photo: {without_photos} ({without_photos/total*100:.1f}%)")
        print(f"{'='*60}")
    except ImportError:
        # Fallback if pandas not available
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            total = sum(1 for row in reader)
        print(f"\n✅ Processed {total} inquiries. Open the CSV file to see details.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the photo extraction."""
    
    # Load API key from .env file
    load_dotenv()
    global PERSONA_API_KEY
    PERSONA_API_KEY = os.getenv("PERSONA_API_KEY")
    
    if not PERSONA_API_KEY:
        print("❌ ERROR: Missing PERSONA_API_KEY in .env file")
        print("\nPlease create a .env file in the same folder as this script with:")
        print("PERSONA_API_KEY=your_api_key_here")
        return
    
    # Create data folder if it doesn't exist
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"📁 Using data folder: {DATA_FOLDER}")
    print(f"   (You can find your input and output files here)\n")
    
    # Set up file paths
    input_csv = DATA_FOLDER / INPUT_CSV_NAME
    output_csv = DATA_FOLDER / OUTPUT_CSV_NAME
    
    # Check if input file exists
    if not input_csv.exists():
        print(f"❌ ERROR: Input file not found: {input_csv}")
        print(f"\nPlease make sure your input CSV file is named '{INPUT_CSV_NAME}'")
        print(f"and is placed in: {DATA_FOLDER}")
        print(f"\nThe CSV file should have an 'inquiry_id' column.")
        return
    
    print(f"📥 Reading from: {input_csv.name}")
    print(f"📤 Will save to: {output_csv.name}\n")
    
    # Run the extraction
    try:
        extract_front_photos_from_csv(str(input_csv), str(output_csv))
        print(f"\n💡 TIP: You can now open '{output_csv.name}' in Google Sheets!")
        print(f"   File location: {output_csv}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

