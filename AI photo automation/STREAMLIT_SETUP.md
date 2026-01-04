# Streamlit Setup Instructions

## Quick Start

Streamlit provides **much better file upload handling** than Gradio and avoids the `ClientDisconnect` errors.

### To Run Streamlit:

1. **Make sure Streamlit is installed:**
   ```bash
   pip install streamlit
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run app_streamlit.py
   ```

3. **Or from Jupyter notebook:**
   ```python
   !streamlit run app_streamlit.py
   ```

### Benefits of Streamlit:

- ✅ **Better file upload handling** - No more ClientDisconnect errors
- ✅ **Handles multiple batches** - Upload second batch without issues
- ✅ **More reliable** - Files are processed more reliably
- ✅ **Better progress tracking** - Real-time progress updates
- ✅ **Cleaner UI** - Modern, responsive interface

### Features:

- Market photos from CSV (same as Gradio)
- File uploads (much more reliable than Gradio)
- UX Focus slider (0-100%)
- Real-time progress tracking
- Stop button
- Results display (summary, prompts, dataframes)

The Streamlit app (`app_streamlit.py`) contains all the same functionality as the Gradio version, but with better file upload handling!
