"""Classical baselines on FMA's precomputed librosa features (features.csv).
Official split:  python src/baselines.py --meta data/fma_metadata
Random split:    python src/baselines.py --meta data/fma_metadata --ids_dir data/processed_random --out results_random
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

ap = argparse.ArgumentParser()
ap.add_argument("--meta", required=True)
ap.add_argument("--out", default="results")
ap.add_argument("--ids_dir", default=None, help="use the split stored in this processed dir (e.g. data/processed_random)")
a = ap.parse_args()

tracks = pd.read_csv(Path(a.meta) / "tracks.csv", index_col=0, header=[0, 1])
feats = pd.read_csv(Path(a.meta) / "features.csv", index_col=0, header=[0, 1, 2])
small = tracks[tracks["set", "subset"] == "small"]
small = small[small["track", "genre_top"].notna()]


def get(split):
    if a.ids_dir:
        df = small.loc[np.load(Path(a.ids_dir) / f"{split}_ids.npy")]
    else:
        df = small[small["set", "split"] == split]
    return feats.loc[df.index].fillna(0).values, df["track", "genre_top"].values


Xtr, ytr = get("training"); Xte, yte = get("test")
print("train/test sizes:", len(Xtr), len(Xte))
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

models = {
    "KNN": KNeighborsClassifier(),
    "LogReg": LogisticRegression(max_iter=1000),
    "MLP": MLPClassifier(max_iter=300, random_state=0),
    "LinearSVM": LinearSVC(dual=False),
}
res = {}
for n, m in models.items():
    m.fit(Xtr, ytr); res[n] = float(accuracy_score(yte, m.predict(Xte)))
    print(f"{n}: {res[n]:.4f}", flush=True)
Path(a.out).mkdir(exist_ok=True)
json.dump(res, open(Path(a.out) / "baselines.json", "w"), indent=2)
