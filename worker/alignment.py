import re
import torchaudio


def get_timestamps(audio_path: str, text: str, language: str = "id") -> list:
    waveform, sr = torchaudio.load(audio_path)
    duration = waveform.shape[1] / sr

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    total_chars = sum(len(s) for s in sentences)

    if total_chars == 0:
        return []

    segments = []
    current_time = 0.0

    for sentence in sentences:
        sent_chars = len(sentence)
        sent_duration = (sent_chars / total_chars) * duration

        words = sentence.split()
        words_total = sum(len(w) for w in words)

        word_segments = []
        word_time = current_time
        for word in words:
            word_duration = (len(word) / words_total) * sent_duration if words_total > 0 else 0
            word_segments.append({
                "word": word,
                "start": round(word_time, 3),
                "end": round(word_time + word_duration, 3),
                "score": 1.0,
            })
            word_time += word_duration

        segments.append({
            "start": round(current_time, 3),
            "end": round(current_time + sent_duration, 3),
            "text": sentence,
            "words": word_segments,
        })

        current_time += sent_duration

    return segments
