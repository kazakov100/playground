# Persona Photo Extractor - Simple Guide

This tool extracts front photo URLs from Persona inquiries and saves them to a CSV file that you can easily sync with Google Sheets.

## Quick Start

### 1. Setup (One-time)

1. **Install Python** (if not already installed)
   - Download from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

2. **Install required packages**
   - Open Terminal (Mac) or Command Prompt (Windows)
   - Navigate to this folder
   - Run: `pip install requests python-dotenv`

3. **Set up API key**
   - Create a file named `.env` in this folder
   - Add this line (replace with your actual API key):
     ```
     PERSONA_API_KEY=your_actual_api_key_here
     ```

### 2. Prepare Your Input File

1. **Create your input CSV file**
   - Name it: `Persona_inquiries.csv`
   - Place it in: `~/Desktop/Persona_Data/` (on Mac) or `C:\Users\YourName\Desktop\Persona_Data\` (on Windows)
   - The CSV must have a column named `inquiry_id` (or `inquiry id`, `inquiryid`, `id`, or `inquiry`)
   - Optional: Include a `file_name` column (or `File name`, `filename`, `file_name`)

   **Example CSV format:**
   ```csv
   inquiry_id,File name
   inq_zyJWx11B4uz2Pn9DnQCEGh9f2BpS,56170112
   inq_wfRJoBsxBDkb4iVM9j5j8JRLpUsE,56170113
   ```

### 3. Run the Script

1. **Double-click** `extract_persona_photos.py` OR
2. **Run from Terminal/Command Prompt:**
   ```bash
   python extract_persona_photos.py
   ```

### 4. Find Your Results

- The output file `Persona_front_photos.csv` will be created in the same folder as your input file
- **Location:** `~/Desktop/Persona_Data/Persona_front_photos.csv` (Mac) or `C:\Users\YourName\Desktop\Persona_Data\Persona_front_photos.csv` (Windows)

### 5. Sync with Google Sheets

1. Open Google Sheets
2. Go to **File → Import**
3. Upload your `Persona_front_photos.csv` file
4. Or use Google Sheets' built-in import from file feature

## Output File Format

The output CSV will have these columns:
- `file_name` - The file name from your input (or inquiry_id if not provided)
- `inquiry_id` - The Persona inquiry ID
- `front_photo_url` - The URL to the front photo (if found)
- `status` - Status message (OK, error message, etc.)

## Customizing File Locations

If you want to change where files are stored, edit the `extract_persona_photos.py` file and look for this section at the top:

```python
# Folder where input and output CSV files will be stored
DATA_FOLDER = Path.home() / "Desktop" / "Persona_Data"

# Input CSV file name
INPUT_CSV_NAME = "Persona_inquiries.csv"

# Output CSV file name
OUTPUT_CSV_NAME = "Persona_front_photos.csv"
```

Change these to match your preferences!

## Troubleshooting

### "File not found" error
- Make sure your input CSV is named correctly and in the right folder
- Check that the folder `Persona_Data` exists on your Desktop

### "Missing PERSONA_API_KEY" error
- Make sure you created a `.env` file in the same folder as the script
- Check that the API key is correct (no extra spaces)

### "No module named 'requests'" error
- Run: `pip install requests python-dotenv`

## Need Help?

If you encounter any issues, check:
1. Your input CSV file format
2. Your API key in the `.env` file
3. That all required packages are installed




