# “Error installing requirements” — get the *real* error

The message Streamlit shows in the UI is **generic**. The actual reason is always in the **build logs**.

## What to do

1. Open your app on [Streamlit Community Cloud](https://share.streamlit.io).
2. Click **Manage app** (or **⋮** → **Manage app**).
3. Open the **Logs** / **Terminal** tab (wording varies).
4. Scroll to the **bottom** of the log and look for lines containing:
   - `ERROR`
   - `Failed`
   - `Could not find`
   - `No matching distribution`
   - `ResolutionImpossible`
5. Copy **10–30 lines** around the first failure (including the `pip` or `uv` command if shown) and paste it into a message / ticket.

Without that snippet, nobody can tell whether the problem is a **bad package name**, **Python version**, **network**, or **conflicting files** in the repo.

## Quick checks on your repo

| Check | Why |
|--------|-----|
| **Main file path** is **`app.py`** (repo root only) | **Never** use `AI photo automation/app.py` — **spaces in the path** break the installer; logs show `Failed to parse: automation/requirements.txt` because the path is split wrong. |
| **No `uv.lock` at repo root** on GitHub | If present, Cloud may use `uv` and try to install the whole monorepo. Remove with `git rm --cached uv.lock` if it was committed. |
| **No `pyproject.toml` at repo root** on GitHub | This repo uses `playground_pyproject.toml` instead so Cloud falls back to `requirements.txt`. |

### If your log says `automation/requirements.txt` or `Invalid requirement: 'automation/requirements.txt'`

That means the **Main file path** was set to something like **`AI photo automation/app.py`**. Cloud’s dependency step does not handle that space correctly.

**Fix:** In app settings, set **Main file path** to **`app.py`** (the file at the **root** of the repo). Reboot the app.

## Official docs

- [App dependencies (requirements, packages)](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [File organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
