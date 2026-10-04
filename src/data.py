"""Convert FMA-small mp3s into fixed-size log-mel spectrograms.

Usage:
  python src/data.py --meta data/fma_metadata --audio data/fma_small --out data/processed
Uses FMA's official split (training / validation / test = 80/10/10).
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, librosa
from joblib import Parallel, delayed
from tqdm import tqdm

SR, N_FFT, HOP, N_MELS, FRAMES = 22050, 2048, 1024, 128, 640  # ~29.7 s


def audio_to_mel(y):
    """Waveform -> log-mel (N_MELS, T). Shared with the real-time demo."""
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(m, ref=1.0).astype(np.float32)


def fix_length(m):
    if m.shape[1] < FRAMES:
        m = np.pad(m, ((0, 0), (0, FRAMES - m.shape[1])), constant_values=m.min())
    return m[:, :FRAMES]


def load_one(path):
    try:
        y, _ = librosa.load(path, sr=SR, duration=30)
        return fix_length(audio_to_mel(y)).astype(np.float16)
    except Exception:
        return None  # a few FMA mp3s are corrupt; they get skipped


def track_path(audio_dir, tid):
    s = f"{tid:06d}"
    return str(Path(audio_dir) / s[:3] / f"{s}.mp3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()

    tracks = pd.read_csv(Path(a.meta) / "tracks.csv", index_col=0, header=[0, 1])
    small = tracks[tracks["set", "subset"] == "small"]
    classes = sorted(small["track", "genre_top"].dropna().unique())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(classes, open(out / "classes.json", "w"))

    for split in ["training", "validation", "test"]:
        df = small[small["set", "split"] == split]
        ids = df.index.tolist()
        mels = Parallel(n_jobs=a.jobs)(
            delayed(load_one)(track_path(a.audio, t)) for t in tqdm(ids, desc=split))
        keep = [i for i, m in enumerate(mels) if m is not None]
        X = np.stack([mels[i] for i in keep])
        y = np.array([classes.index(df["track", "genre_top"].iloc[i]) for i in keep])
        np.save(out / f"{split}_X.npy", X); np.save(out / f"{split}_y.npy", y)
        np.save(out / f"{split}_ids.npy", np.array([ids[i] for i in keep]))
        print(split, X.shape, f"skipped {len(ids) - len(keep)}")


if __name__ == "__main__":
    main()
