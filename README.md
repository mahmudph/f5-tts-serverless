# F5-TTS RunPod Serverless

RunPod serverless worker for text-to-speech inference using [F5-TTS](https://github.com/SWivid/F5-TTS) v1 Base model. Output audio is uploaded to Cloudflare R2 and returned as a public URL.

## Features

- Runs on RunPod Serverless GPU workers.
- Uses `F5TTS_v1_Base` (hardcoded).
- Accepts a reference audio URL for voice cloning.
- Returns generated audio as a public R2 URL (avoids RunPod payload limits).
- Supports `wav` and `mp3` output formats.
- Builds Docker image via GitHub Actions (manual trigger) and pushes to GHCR.

## API

### Input

```json
{
  "input": {
    "text": "Text to speak",
    "ref_audio": "https://example.com/reference-voice.wav",
    "ref_text": "Reference audio transcription",
    "output_format": "wav"
  }
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `text` | Yes | Text to synthesize. |
| `ref_audio` | Yes | Public URL to a reference audio file. |
| `ref_text` | No | Transcription of the reference audio. If empty, the model transcribes it automatically (uses extra VRAM). |
| `output_format` | No | `wav` (default) or `mp3`. |

### Output

```json
{
  "audio_url": "https://pub-xxx.r2.dev/tts-output/uuid.wav",
  "model_used": "F5TTS_v1_Base",
  "sample_rate": 24000,
  "duration_seconds": 5.2
}
```

On failure:

```json
{
  "error": "error message"
}
```

## Deploy

### 1. Build with GitHub Actions

This repo includes a GitHub Actions workflow that builds the Docker image and pushes it to GitHub Container Registry (GHCR). The workflow is triggered **manually** via `workflow_dispatch`.

Go to **Actions** → select **Build and Push Docker Image** → click **Run workflow**.

### 2. Make GHCR package public

After the first successful build:

1. Go to GitHub → **Packages** → `<your-image>`.
2. Open **Package settings** → **Manage visibility**.
3. Select **Public** and save.

If you keep it private, add a **Container Registry Auth** in RunPod with a GitHub PAT that has `read:packages` scope.

### 3. Create RunPod endpoint

1. Open [RunPod Console](https://console.runpod.io).
2. Create a new **Serverless Endpoint**.
3. Use the image:
   ```text
   ghcr.io/<username>/<repo>:latest
   ```
4. Select a GPU (e.g., RTX 3090 / A4000 / A5000 or higher).
5. Set the environment variables below.

## Environment Variables

Configure these in your RunPod endpoint settings:

| Variable | RunPod Type | Description |
|----------|-------------|-------------|
| `R2_ACCOUNT_ID` | Environment Variable | Cloudflare account ID. |
| `R2_ACCESS_KEY_ID` | Environment Variable or Secrets | R2 access key. |
| `R2_SECRET_ACCESS_KEY` | **Secrets** | R2 secret key. |
| `R2_BUCKET_NAME` | Environment Variable | Target R2 bucket. |
| `R2_PUBLIC_DOMAIN` | Environment Variable | Optional public domain, e.g. `https://pub-xxx.r2.dev`. |

> **Note:** Always store `R2_SECRET_ACCESS_KEY` as a RunPod Secret, not a plain environment variable.

## Local Testing (Optional)

If you have a GPU locally and want to test before deploying:

```bash
export $(cat .env.example | xargs)
python worker/handler.py --test_input '{"input": {"text": "hello", "ref_audio": "https://..."}}'
```

## Project Structure

```
worker/
  handler.py       # RunPod serverless entrypoint
  inference.py     # F5-TTS model loading and inference
  storage.py       # R2 upload and reference audio download
  Dockerfile       # CUDA 12.1 runtime image
  requirements.txt # Python dependencies
```

## Notes

- The model is downloaded from HuggingFace on the first cold start.
- Reference audio must be accessible via a public URL.
- Output is uploaded to R2 to avoid RunPod's payload size limits.
