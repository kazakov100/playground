# Setup Instructions for Teammate

## Files to Share

Share these files with your teammate:

1. **`extract_persona_photos.py`** - The main script
2. **`README_PERSONA_PHOTOS.md`** - Detailed instructions
3. **`.env.example`** - Template for API key (they'll need to create their own `.env` file)

## Quick Setup Checklist

Your teammate needs to:

- [ ] Install Python (if not already installed)
- [ ] Install packages: `pip install requests python-dotenv`
- [ ] Create `.env` file with their `PERSONA_API_KEY`
- [ ] Create input CSV file: `Persona_inquiries.csv` in `~/Desktop/Persona_Data/`
- [ ] Run the script: `python extract_persona_photos.py`
- [ ] Find output in: `~/Desktop/Persona_Data/Persona_front_photos.csv`

## File Locations

**Input file location:**
- Mac: `~/Desktop/Persona_Data/Persona_inquiries.csv`
- Windows: `C:\Users\[Username]\Desktop\Persona_Data\Persona_inquiries.csv`

**Output file location:**
- Mac: `~/Desktop/Persona_Data/Persona_front_photos.csv`
- Windows: `C:\Users\[Username]\Desktop\Persona_Data\Persona_front_photos.csv`

These locations are easy to find and perfect for Google Sheets import!

## Customizing Paths

If your teammate wants to use a different folder, they can edit the top of `extract_persona_photos.py`:

```python
DATA_FOLDER = Path.home() / "Desktop" / "Persona_Data"  # Change this line
```

They can use any path, for example:
- `Path.home() / "Documents" / "Persona"` 
- `Path("/path/to/shared/folder")`
- `Path("C:/Users/Shared/Persona")` (Windows)

## Notes

- The script automatically creates the `Persona_Data` folder if it doesn't exist
- All files (input and output) are in the same folder for easy access
- The output CSV is ready to import directly into Google Sheets
- The script provides clear progress messages and error handling




