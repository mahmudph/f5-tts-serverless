# AGENTS.md

## Project state

- RunPod serverless worker for TTS inference using F5-TTS v1 Base model.
- Output audio is uploaded to Cloudflare R2 and returned as a public URL.
- No local execution required; build the Docker image and deploy/test on RunPod.
- Python 3.10 inside `nvidia/cuda:12.1.0-runtime-ubuntu22.04` base image.

## Project layout

- `worker/handler.py` — RunPod serverless entrypoint.
- `worker/inference.py` — Model loading and TTS synthesis via `f5-tts` high-level API.
- `worker/storage.py` — Download reference audio and upload output to R2.
- `worker/Dockerfile` — CUDA 12.1 image for GPU inference.
- `worker/requirements.txt` — Python dependencies.
- `.env.example` — Required environment variables.
- `test_input.json` — Sample payload for local testing.

## Build & deploy

### Option A: Build manually with Docker

```bash
docker build -t <your-image>:latest worker/
docker push <your-image>:latest
```

### Option B: Build with GitHub Actions (recommended)

A workflow is provided at `.github/workflows/docker-build.yml`. It builds and pushes the image to **GitHub Container Registry (GHCR)** when triggered manually via `workflow_dispatch`.

1. Push this repo to GitHub.
2. Go to **Actions** → select **Build and Push Docker Image** → click **Run workflow**.
3. After the first successful push, make the GHCR package public:
   - GitHub → **Packages** → `f5-fts` → **Package settings** → **Manage visibility** → **Public**.
4. In RunPod, use the image:
   ```text
   ghcr.io/<username>/<repo>:latest
   ```

If the GHCR package is kept private, add a **Container Registry Auth** in RunPod with a GitHub PAT that has `read:packages` scope.

### Deploy to RunPod

Create a RunPod Serverless Endpoint, point it to the image, and set the env vars from `.env.example`.

## Environment variables

`.env.example` is only a template; the worker reads env vars that are injected by RunPod at runtime.

### RunPod Console

In the endpoint settings, add the variables below:

| Variable | Where to set | Notes |
|----------|--------------|-------|
| `R2_ACCOUNT_ID` | Environment Variables | Cloudflare account ID. |
| `R2_ACCESS_KEY_ID` | Environment Variables or Secrets | R2 access key. |
| `R2_SECRET_ACCESS_KEY` | **Secrets** | R2 secret key; always store sensitive values as Secrets. |
| `R2_BUCKET_NAME` | Environment Variables | Target bucket for generated audio. |
| `R2_PUBLIC_DOMAIN` | Environment Variables | Optional custom public domain, e.g. `https://pub-xxx.r2.dev`. |

- **Environment Variables** are visible in the RunPod UI and are fine for non-sensitive config.
- **Secrets** are encrypted and hidden; use them for `R2_SECRET_ACCESS_KEY`.
- Never commit a real `.env` file to git.

## API input

Required: `text`, `ref_audio` (public URL).
Optional:
- `ref_text` — transcription of reference audio (auto-transcribed if empty, uses extra VRAM).
- `output_format` — `wav` (default) or `mp3`.

## API output

```json
{
  "audio_url": "https://...",
  "model_used": "F5TTS_v1_Base",
  "sample_rate": 24000,
  "duration_seconds": 5.2
}
```

On failure, handler returns `{"error": "..."}`.

## Notes

- Payload limits are avoided by returning R2 URLs instead of base64 audio.
- Model and vocoder are loaded once per worker outside the handler.
- Reference audio is downloaded from a public URL; base64 upload is not supported.
- First cold start downloads the HuggingFace model; subsequent workers on the same host reuse cache.
- Keep `HF_HOME` as `/root/.cache/huggingface` (default) unless you attach a network volume.

## Things to check before editing

- Do not pin a torch CPU build; `requirements.txt` uses `+cu121` with the PyTorch CUDA 12.1 index.
- `f5-tts` is imported via `from f5_tts.api import F5TTS`; avoid calling low-level `load_model`/`infer_process` directly unless you understand the config/ckpt plumbing.
- R2 env vars must all be present at runtime; missing credentials will fail the job with a clear error.
