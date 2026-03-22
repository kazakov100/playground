# AI photo automation — CSV prompt evaluator

Streamlit app to evaluate vision prompts against a CSV of images (`image_url`, `market_name`, `ground_truth`).

## Run locally

```bash
cd "AI photo automation"
pip install -r requirements.txt
export OPENROUTER_API_KEY=...   # or use a `.env` next to this folder / repo root
streamlit run app.py
```

## Layout

| Path | Role |
|------|------|
| `app.py` | Streamlit UI (main entry) |
| `core/openrouter.py` | OpenRouter API calls, image → data URL, classification |
| `core/optimizer.py` | Optional sequential evaluation loop (not wired into `app.py`) |
| `app_streamlit.py` | Alternate / older Streamlit UI |

## Deploy

See `DEPLOY_INSTRUCTIONS.md` (Streamlit Community Cloud).

## Config

- **Model / temperature:** sidebar in the app.
- **System prompt:** `SYSTEM_PROMPT` in `app.py`.
- **API key:** `OPENROUTER_API_KEY` (env or `.env`).
