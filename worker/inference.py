import torch
from f5_tts.api import F5TTS
from huggingface_hub import hf_hub_download

MODELS = {}

CUSTOM_MODELS = {
    "F5TTS-INDO-V2": {
        "config": "F5TTS_v1_Base",
        "repo_id": "Eempostor/F5-TTS-INDO-FINETUNE-V2",
        "ckpt_file": "f5_tts_indo_v2.pt",
        "vocab_file": "vocab.txt",
    },
}


def _download_custom_model_files(cfg: dict) -> tuple[str, str]:
    ckpt = hf_hub_download(repo_id=cfg["repo_id"], filename=cfg["ckpt_file"])
    vocab = hf_hub_download(repo_id=cfg["repo_id"], filename=cfg["vocab_file"])
    return ckpt, vocab


def load_tts_model(model_name: str = "F5TTS_v1_Base"):
    if model_name not in MODELS:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_name in CUSTOM_MODELS:
            cfg = CUSTOM_MODELS[model_name]
            ckpt_file, vocab_file = _download_custom_model_files(cfg)
            MODELS[model_name] = F5TTS(
                model=cfg["config"],
                ckpt_file=ckpt_file,
                vocab_file=vocab_file,
                device=device,
            )
        else:
            MODELS[model_name] = F5TTS(model=model_name, device=device)
    return MODELS[model_name]


def synthesize(
    text: str,
    ref_audio_path: str,
    ref_text: str = "",
    model_name: str = "F5TTS_v1_Base",
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
