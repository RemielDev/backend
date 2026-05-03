# Bloom AI — Backend

FastAPI service that analyzes and moderates chat from the Bloom Roblox game.

## Run locally

```powershell
pip install -r requirements.txt
python -m uvicorn api.app:app --reload --port 8000
```

Then open <http://localhost:8000/health>.

## Env vars

Set these in `.env` locally and in Railway's **Variables** tab for deploy:

- `GOOGLE_API_KEY` — Gemini key (required, server won't boot without it)
- `BLOOM_API_KEY` — shared secret with Roblox `Config.luau`. Must match.
- `SUPABASE_URL`, `SUPABASE_KEY` — database
- `ROBLOX_PLATFORM_API_KEY`, `ROBLOX_UNIVERSE_ID` — Roblox Open Cloud
- `HF_TOKEN` — Hugging Face inference

## Deploy

Push to Railway. It runs the `Procfile` and injects env vars from the Variables tab. After deploy, update `API_URL` in the game's `Config.luau` to the Railway URL.
