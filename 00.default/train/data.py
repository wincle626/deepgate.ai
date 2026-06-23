import os
import logging

import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
NUM_SAMPLES = 16000  # 1 second at 16 kHz

_mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=10,
    log_mels=True,
    melkwargs=dict(
        n_fft=512,
        hop_length=320,
        n_mels=40,
        f_min=20.0,
        f_max=4000.0,
        center=False,
        power=1.0,
    ),
)

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def load_wav(path):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    waveform = torch.from_numpy(data).unsqueeze(0)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    return waveform


def pad_or_trim(waveform):
    if waveform.shape[1] < NUM_SAMPLES:
        waveform = F.pad(waveform, (0, NUM_SAMPLES - waveform.shape[1]))
    elif waveform.shape[1] > NUM_SAMPLES:
        waveform = waveform[:, :NUM_SAMPLES]
    return waveform


def compute_mfcc(waveform):
    waveform = pad_or_trim(waveform)
    mfcc = _mfcc_transform(waveform)
    mfcc = mfcc.permute(0, 2, 1)
    return mfcc.unsqueeze(1)


def load_from_wavs(data_dir, device="cpu"):
    """Load class-folder-structured WAVs into a TensorDataset.

    Expected layout:
        data_dir/
            class_a/file1.wav, file2.wav, ...
            class_b/file3.wav, ...
    """
    data_dir = os.path.expanduser(data_dir)
    class_names = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    x_list, y_list = [], []
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        files = sorted(
            f for f in os.listdir(class_dir)
            if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
        )
        for fname in files:
            waveform = load_wav(os.path.join(class_dir, fname))
            x_list.append(compute_mfcc(waveform))
            y_list.append(class_idx)

    x = torch.cat(x_list, dim=0)
    y = torch.tensor(y_list, dtype=torch.long)
    return torch.utils.data.TensorDataset(x.to(device), y.half().to(device))

KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
CLASSES = ["silence", "unknown"] + KEYWORDS  # 12 classes
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASSES)}


def _map_label(word):
    return LABEL_TO_IDX.get(word, LABEL_TO_IDX["unknown"])


def _get_sc_path(root):
    return os.path.join(root, "SpeechCommands", "speech_commands_v0.02")


def _ensure_downloaded(root):
    sc_path = _get_sc_path(root)
    if not os.path.isdir(sc_path):
        logger.info("Downloading Speech Commands v2...")
        os.makedirs(root, exist_ok=True)  # torchaudio writes the archive into root but won't create it
        torchaudio.datasets.SPEECHCOMMANDS(root, download=True, subset="testing")
    return sc_path


def _get_split_files(sc_path, subset):
    all_files = []
    for word_dir in sorted(os.listdir(sc_path)):
        word_path = os.path.join(sc_path, word_dir)
        if not os.path.isdir(word_path) or word_dir.startswith("_"):
            continue
        for fname in sorted(os.listdir(word_path)):
            if fname.endswith(".wav"):
                rel_path = os.path.join(word_dir, fname)
                all_files.append((os.path.join(word_path, fname), word_dir, rel_path))

    val_list, test_list = set(), set()
    for name, target in (("validation_list.txt", val_list), ("testing_list.txt", test_list)):
        p = os.path.join(sc_path, name)
        if os.path.exists(p):
            target.update(line.strip() for line in open(p))

    result = []
    for abs_path, label, rel_path in all_files:
        if subset == "testing" and rel_path in test_list:
            result.append((abs_path, label))
        elif subset == "validation" and rel_path in val_list:
            result.append((abs_path, label))
        elif subset in ("training", "train+val") and rel_path not in test_list:
            if subset == "train+val" or rel_path not in val_list:
                result.append((abs_path, label))
    return result


def _load_background_noise(sc_path):
    noise_dir = os.path.join(sc_path, "_background_noise_")
    if not os.path.isdir(noise_dir):
        return None
    chunks = [load_wav(os.path.join(noise_dir, f))[0]
              for f in sorted(os.listdir(noise_dir)) if f.endswith(".wav")]
    return torch.cat(chunks) if chunks else None


def _generate_silence_mfccs(sc_path, num_samples):
    noise = _load_background_noise(sc_path)
    if noise is None or len(noise) < NUM_SAMPLES:
        mfccs = compute_mfcc(torch.zeros(1, NUM_SAMPLES)).expand(num_samples, -1, -1, -1)
        return mfccs.clone()
    x_list = []
    for _ in range(num_samples):
        start = torch.randint(0, len(noise) - NUM_SAMPLES, (1,)).item()
        seg = noise[start:start + NUM_SAMPLES].unsqueeze(0)
        x_list.append(compute_mfcc(seg))
    return torch.cat(x_list, dim=0)


def _build_and_cache(root, subset):
    sc_path = _ensure_downloaded(root)
    files = _get_split_files(sc_path, subset)

    x_list, y_list, keyword_count = [], [], 0
    for filepath, label in files:
        waveform = load_wav(filepath)
        x_list.append(compute_mfcc(waveform))
        y_list.append(_map_label(label))
        if label in KEYWORDS:
            keyword_count += 1

    num_silence = keyword_count // len(KEYWORDS) if keyword_count else 100
    x_list.append(_generate_silence_mfccs(sc_path, num_silence))
    y_list.extend([LABEL_TO_IDX["silence"]] * num_silence)

    x = torch.cat(x_list, dim=0)
    y = torch.tensor(y_list, dtype=torch.long)

    cache_dir = os.path.join(root, "speech_commands_cache")
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(x, os.path.join(cache_dir, f"{subset}_x.pt"))
    torch.save(y, os.path.join(cache_dir, f"{subset}_y.pt"))
    return x, y


def load_speech_commands(root="./data", subset="training", device="cpu"):
    root = os.path.expanduser(root)
    cache_dir = os.path.join(root, "speech_commands_cache")
    x_path = os.path.join(cache_dir, f"{subset}_x.pt")
    y_path = os.path.join(cache_dir, f"{subset}_y.pt")
    if os.path.exists(x_path) and os.path.exists(y_path):
        x = torch.load(x_path, weights_only=True)
        y = torch.load(y_path, weights_only=True)
    else:
        x, y = _build_and_cache(root, subset)
    return torch.utils.data.TensorDataset(x.to(device), y.half().to(device))
