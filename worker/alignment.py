import nltk
import whisperx
import torch

ALIGN_MODELS = {}


def _load_align_model(language: str = "id"):
    key = f"align_{language}"
    if key not in ALIGN_MODELS:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        nltk.download("punkt", quiet=True)
        model, metadata = whisperx.load_align_model(language_code=language, device=device)
        ALIGN_MODELS[key] = (model, metadata, language, device)
    return ALIGN_MODELS[key]


def get_timestamps(audio_path: str, text: str, language: str = "id") -> list:
    model, metadata, lang, device = _load_align_model(language)
    sentences = nltk.sent_tokenize(text)
    result = whisperx.align(sentences, model, metadata, audio_path, device)
    return result["segments"]
