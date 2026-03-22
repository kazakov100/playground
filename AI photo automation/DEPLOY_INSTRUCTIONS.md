# Deploying the Streamlit app (shareable link)

Use **Streamlit Community Cloud** for a public URL.

## 1. Push your repo to GitHub

Ensure the app includes:

- `AI photo automation/app.py`
- `AI photo automation/core/`
- `AI photo automation/requirements.txt` (or root `requirements.txt`)

## 2. Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **Create app** → select your repo and branch.
3. **Main file path:** `AI photo automation/app.py` (adjust if your paths differ).
4. **Secrets** (Settings → Secrets):

   ```toml
   OPENROUTER_API_KEY = "your-key-here"
   ```

5. Deploy. You’ll get a URL like `https://<name>.streamlit.app`.

Do **not** commit `.env`; use Streamlit Secrets only.

## If “Error installing requirements” appears

**Cause (common in monorepos):** a file named **`uv.lock` in the repo root** makes Community Cloud use **`uv`** and install **everything** in that lock (often huge and slow). This repo keeps the lock as **`playground_uv.lock`** instead so Cloud falls back to **`requirements.txt`**.

**If you still see errors:**

1. Open **Manage app → Logs** and copy the **first pip/uv error line**.
2. Confirm **Main file path** is exactly `AI photo automation/app.py` (forward slashes).
3. In **Advanced settings**, pick **Python 3.11** if available.

## Local run

```bash
cd "AI photo automation"
pip install -r requirements.txt
streamlit run app.py
```
