# ai-review-tester
Temporary repo for internal Bird AI photo review testing.  
Shared privately for engineering review.

**Streamlit evaluator:** see `AI photo automation/README.md` — from repo root: `streamlit run app.py`.

**Python project metadata:** the full monorepo definition is in **`playground_pyproject.toml`** (not `pyproject.toml` at the root) so **Streamlit Community Cloud** only sees **`requirements.txt`** for installs. The lockfile is **`playground_uv.lock`** (never commit **`uv.lock`** at repo root — it overrides `requirements.txt`). For `uv` locally, use `uv sync --project playground_pyproject.toml` or rename the file when needed.

**Deploy issues:** see `AI photo automation/DEPLOY_INSTRUCTIONS.md` and `STREAMLIT_CLOUD_LOGS.md`.
