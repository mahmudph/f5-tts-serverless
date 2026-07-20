import re
import torch
import torchaudio

ALIGN_MODELS = {}

def _load_align_model(language="id"):
    key = "wav2vec2"
    if key not in ALIGN_MODELS:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model().to(device)
        labels = list(bundle.get_labels())
        ALIGN_MODELS[key] = (model, labels, bundle, device)
    return ALIGN_MODELS[key]


def _tokenize(text, label_map):
    tokens = []
    for char in text.upper():
        if char == " " and "|" in label_map:
            tokens.append(label_map["|"])
        elif char in label_map:
            tokens.append(label_map[char])
    return tokens


def _trellis(emissions, tokens, blank):
    T, N = emissions.shape
    L = len(tokens)
    if L == 0:
        return None

    S = 2 * L + 1
    trellis = torch.full((T, S), float("-inf"))

    trellis[0, 0] = emissions[0, blank]
    if S > 1:
        trellis[0, 1] = emissions[0, tokens[0]]

    for t in range(1, T):
        max_s = min(S, 2 * t + 1)
        min_s = max(0, 2 * (t - T + L))

        for s in range(min_s, max_s):
            val = trellis[t - 1, s]
            if s > 0:
                val = torch.logaddexp(val, trellis[t - 1, s - 1])
            if s > 1:
                curr = tokens[(s - 1) // 2]
                prev = tokens[(s - 3) // 2]
                if curr != prev:
                    val = torch.logaddexp(val, trellis[t - 1, s - 2])

            label = blank if s % 2 == 0 else tokens[(s - 1) // 2]
            trellis[t, s] = val + emissions[t, label]

    return trellis


def _backtrack(trellis, tokens, blank):
    T, S = trellis.shape
    s = S - 1 if (S - 1) % 2 == 1 else S - 2
    path = [s]

    for t in range(T - 1, 0, -1):
        candidates = [c for c in [s, s - 1, s - 2] if 0 <= c < S]
        s = max(candidates, key=lambda c: trellis[t - 1, c])
        path.append(s)

    path.reverse()
    return path


def _word_times(path, tokens, labels, blank, time_per_frame):
    words = []
    current = []
    word_start = None

    for t, s in enumerate(path):
        if s % 2 == 0:
            continue

        char_idx = tokens[(s - 1) // 2]
        label = labels[char_idx]

        if label == "|":
            if current:
                words.append((word_start, t))
                current = []
                word_start = None
            continue

        if word_start is None:
            word_start = t
        current.append(char_idx)

    if current:
        words.append((word_start, len(path)))

    result = []
    for start, end in words:
        result.append({
            "start": round(start * time_per_frame, 3),
            "end": round(end * time_per_frame, 3),
        })
    return result


def get_timestamps(audio_path, text, language="id"):
    model, labels, bundle, device = _load_align_model(language)
    blank = 0

    waveform, sr = torchaudio.load(audio_path)
    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)
    waveform = waveform.to(device)

    with torch.inference_mode():
        emissions, _ = model(waveform)
    emissions = torch.log_softmax(emissions[0].cpu(), dim=-1)

    time_per_frame = waveform.shape[1] / bundle.sample_rate / emissions.shape[0]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    orig_words = [s.split() for s in sentences]

    label_map = {l: i for i, l in enumerate(labels)}

    segments = []
    for sentence, orig_word_list in zip(sentences, orig_words):
        tokens = _tokenize(sentence, label_map)
        if len(tokens) < 2:
            continue

        tr = _trellis(emissions, tokens, blank)
        if tr is None:
            continue

        path = _backtrack(tr, tokens, blank)
        wt = _word_times(path, tokens, labels, blank, time_per_frame)

        if not wt:
            continue

        word_segments = []
        for ti, ow in zip(wt, orig_word_list):
            word_segments.append({
                "word": ow,
                "start": ti["start"],
                "end": ti["end"],
                "score": 1.0,
            })

        segments.append({
            "start": word_segments[0]["start"],
            "end": word_segments[-1]["end"],
            "text": sentence,
            "words": word_segments,
        })

    return segments
