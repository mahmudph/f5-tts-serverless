import torch
from f5_tts.api import F5TTS

MODELS = {}


def load_tts_model(model_name: str = "F5TTS_v1_Small"):
    if model_name not in MODELS:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        MODELS[model_name] = F5TTS(model=model_name, device=device)
    return MODELS[model_name]


def synthesize(
    text: str,
    ref_audio_path: str,
    ref_text: str = "",
    model_name: str = "F5TTS_v1_Small",
    output_path: str = "/tmp/output.wav",
    remove_silence: bool = False,
    cfg_strength: float = 2.0,
    nfe_step: int = 32,
    speed: float = 1.0,
    sway_sampling_coef: float = -1.0,
) -> dict:
    model = load_tts_model(model_name)

    wav, sr, spec = model.infer(
        ref_file=ref_audio_path,
        ref_text=ref_text,
        gen_text=text,
        file_wave=output_path,
        remove_silence=remove_silence,
        cfg_strength=cfg_strength,
        nfe_step=nfe_step,
        speed=speed,
        sway_sampling_coef=sway_sampling_coef,
    )

    duration = len(wav) / sr if wav is not None else 0.0

    return {
        "output_path": output_path,
        "sample_rate": sr,
        "duration_seconds": round(duration, 3),
    }
