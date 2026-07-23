import os
import uuid
import runpod
from pydub import AudioSegment
from storage import download_audio, upload_to_r2
from inference import synthesize
from alignment import get_timestamps

DEFAULT_MODEL = "F5TTS_v1_Base"
LANGUAGE_MODEL_MAP = {
    "id": "F5TTS-INDO-V2",
}
ALLOWED_FORMATS = ["wav", "mp3"]


def handler(job):
    job_input = job.get("input", {})

    text = job_input.get("text")
    ref_audio_url = job_input.get("ref_audio")
    ref_text = job_input.get("ref_text", "")
    language = job_input.get("language", "id")
    model_name = job_input.get("model") or LANGUAGE_MODEL_MAP.get(language, DEFAULT_MODEL)
    output_format = job_input.get("output_format", "wav")
    return_timestamps = job_input.get("return_timestamps", False)

    if not text:
        return {"error": "Missing required input: text"}
    if not ref_audio_url:
        return {"error": "Missing required input: ref_audio"}
    if output_format not in ALLOWED_FORMATS:
        return {"error": "output_format must be 'wav' or 'mp3'"}

    job_id = job.get("id", str(uuid.uuid4()))
    tmp_dir = f"/tmp/{job_id}"
    os.makedirs(tmp_dir, exist_ok=True)

    ref_audio_path = os.path.join(tmp_dir, "reference.wav")
    wav_output_path = os.path.join(tmp_dir, "output.wav")
    final_output_path = os.path.join(tmp_dir, f"output.{output_format}")
    content_type = "audio/wav" if output_format == "wav" else "audio/mpeg"

    try:
        download_audio(ref_audio_url, ref_audio_path)

        result = synthesize(
            text=text,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            model_name=model_name,
            output_path=wav_output_path,
            remove_silence=job_input.get("remove_silence", True),
            cfg_strength=job_input.get("cfg_strength", 2.0),
            nfe_step=job_input.get("nfe_step", 32),
            speed=job_input.get("speed", 1.0),
            sway_sampling_coef=job_input.get("sway_sampling_coef", -1.0),
        )

        if return_timestamps:
            segments = get_timestamps(wav_output_path, text, language)
        else:
            segments = None

        if output_format == "mp3":
            AudioSegment.from_wav(wav_output_path).export(final_output_path, format="mp3")
        else:
            final_output_path = wav_output_path

        audio_url = upload_to_r2(final_output_path, content_type=content_type)

        response = {
            "audio_url": audio_url,
            "model_used": model_name,
            "sample_rate": result["sample_rate"],
            "duration_seconds": result["duration_seconds"],
        }
        if return_timestamps and segments is not None:
            response["segments"] = segments

        return response

    except Exception as e:
        return {"error": str(e)}

    finally:
        for f in [ref_audio_path, wav_output_path, final_output_path]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass


runpod.serverless.start({"handler": handler})
