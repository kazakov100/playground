# ai-review-tester
Temporary repo for internal Bird AI photo review testing.  
Shared privately for engineering review.

**Streamlit evaluator:** see `AI photo automation/README.md` — run `streamlit run app.py` from that folder.

**UV / `pyproject.toml`:** the full playground lockfile is stored as `playground_uv.lock` (not `uv.lock`) so **Streamlit Community Cloud** uses the small `requirements.txt` instead of syncing the entire monorepo. To regenerate locally: `uv lock` then rename or commit as needed.
