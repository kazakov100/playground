# Deploying the Streamlit app (shareable link)

The app is **Streamlit**, not Gradio. Use **Streamlit Community Cloud** for a public URL.

## 1. Push your repo to GitHub

Ensure the repo includes `app.py`, `core/`, and a **root** `requirements.txt` (from the `playground` repo root).

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

## Local run

```bash
cd "AI photo automation"
streamlit run app.py
```
