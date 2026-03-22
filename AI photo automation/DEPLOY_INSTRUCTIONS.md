# Deploying the Streamlit app (shareable link)

Use **Streamlit Community Cloud**: [share.streamlit.io](https://share.streamlit.io)

## 1. Push the full repo to GitHub

Include the folder **`AI photo automation/`** and the **root** `requirements.txt`.

## 2. Create the app

1. **Main file path — use one of these (recommended first):**
   - **`app.py`** ← **recommended** (file at **repo root**; no spaces in path)
   - or `AI photo automation/app.py` (must match your repo exactly; use **forward slashes**)

2. **Branch:** e.g. `main`

3. **Secrets** (app → ⚙️ → Secrets):

   ```toml
   OPENROUTER_API_KEY = "your-key-here"
   ```

4. **Deploy**

## If you see “Error installing requirements”

That banner is **not** the real error. See **`STREAMLIT_CLOUD_LOGS.md`** in the repo root for how to copy the **build log** (the lines mentioning `pip`, `uv`, or `ERROR`).

Quick fixes that often help:

1. **Main file path:** `app.py` (at **repository root**).
2. **On GitHub:** no **`uv.lock`** at repo root; no **`pyproject.toml`** at repo root (this project uses **`playground_pyproject.toml`**).
3. **Advanced settings:** try **Python 3.11** if install fails on newer Python.

## Local run (same as Cloud)

From the **repository root**:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or from `AI photo automation/`:

```bash
pip install -r requirements.txt
streamlit run app.py
```

(From that folder, `app.py` refers to the nested file; from repo root, use the **root** `app.py` shim.)
