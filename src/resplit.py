"""Re-split the preprocessed FMA-small clips into a random, class-stratified 70/20/10
split (the protocol described in the SongNet paper) instead of FMA's official split.
Usage: python src/resplit.py --src data/processed --out data/processed_random
"""
import argparse, shutil
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/processed")
ap.add_argument("--out", default="data/processed_random")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

S = ["training", "validation", "test"]
cat = lambda k: np.concatenate([np.load(Path(a.src) / f"{s}_{k}.npy") for s in S])
X, y, ids = cat("X"), cat("y"), cat("ids")
idx = np.arange(len(y))
tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=a.seed)
va, te = train_test_split(rest, test_size=1 / 3, stratify=y[rest], random_state=a.seed)

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for name, sel in zip(S, [tr, va, te]):
    np.save(out / f"{name}_X.npy", X[sel]); np.save(out / f"{name}_y.npy", y[sel])
    np.save(out / f"{name}_ids.npy", ids[sel]); print(name, len(sel))
shutil.copy(Path(a.src) / "classes.json", out / "classes.json")
