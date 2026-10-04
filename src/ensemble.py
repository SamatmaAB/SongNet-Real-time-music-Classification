"""Hybrid ensemble: SongNet run(s) + classical classifiers on FMA's precomputed audio features.
The blend weight is chosen on the VALIDATION set only; test accuracy is reported once at that weight.
Usage:
  python src/ensemble.py --data data/processed_random --meta data/fma_metadata \
      --runs results_random:dense results_random:gru
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from model import SongNet

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--meta", required=True)
ap.add_argument("--runs", nargs="+", required=True, help="<results_dir>:<head> ...")
ap.add_argument("--feat", choices=["mlp", "logreg", "both"], default="both")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--trainval", action="store_true", help="fit the feature models on train+validation")
ap.add_argument("--auto_pool", action="store_true", help="also try SVM and gradient-boosting feature models; pool and weight are chosen on VALIDATION only")
ap.add_argument("--w", type=float, default=None, help="fixed SongNet weight; if omitted it is chosen on validation")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = Path(a.data)
classes = json.load(open(d / "classes.json"))
split = {s: (torch.from_numpy(np.load(d / f"{s}_X.npy")), np.load(d / f"{s}_y.npy"), np.load(d / f"{s}_ids.npy"))
         for s in ["training", "validation", "test"]}


@torch.no_grad()
def songnet_probs(res_dir, head, X):
    norm = json.load(open(Path(res_dir) / f"norm_{head}.json"))
    mean = torch.tensor(np.reshape(norm["mean"], (-1, 1)), dtype=torch.float32, device=dev)
    std = torch.tensor(np.reshape(norm["std"], (-1, 1)), dtype=torch.float32, device=dev)
    m = SongNet(n_classes=len(classes), head=head).to(dev)
    m.load_state_dict(torch.load(Path(res_dir) / f"songnet_{head}.pt", map_location=dev)); m.eval()
    out = []
    for i in range(0, len(X), 64):
        x = (X[i:i + 64].to(dev).float() - mean) / std
        out.append(torch.softmax(m(x), -1).mean(1).cpu())
    return torch.cat(out).numpy()


# --- SongNet part
S = {s: np.mean([songnet_probs(*r.split(":"), split[s][0]) for r in a.runs], axis=0) for s in ["validation", "test"]}

# --- classical feature models (FMA precomputed audio features)
tracks = pd.read_csv(Path(a.meta) / "tracks.csv", index_col=0, header=[0, 1])
feats = pd.read_csv(Path(a.meta) / "features.csv", index_col=0, header=[0, 1, 2])
small = tracks[tracks["set", "subset"] == "small"]


def get(s):
    ids = split[s][2]
    F = feats.loc[ids].fillna(0).values
    lab = np.array([classes.index(g) for g in small.loc[ids]["track", "genre_top"]])
    assert (lab == split[s][1]).all(), "label mismatch between features and spectrogram arrays"
    return F, lab


Ftr, ytr = get("training"); Fva, yva = get("validation"); Fte, yte = get("test")
if a.trainval:
    Ftr, ytr = np.concatenate([Ftr, Fva]), np.concatenate([ytr, yva])
sc = StandardScaler().fit(Ftr); Ftr, Fva, Fte = sc.transform(Ftr), sc.transform(Fva), sc.transform(Fte)
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
assert not (a.auto_pool and a.trainval), "--auto_pool selects on validation, so it cannot be combined with --trainval"


def make(n):
    return {"logreg": LogisticRegression(max_iter=1000),
            "mlp": MLPClassifier(max_iter=300, random_state=a.seed),
            "svm": SVC(kernel="rbf", C=3, probability=True, random_state=a.seed),
            "hgb": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=a.seed)}[n]


names = ["logreg", "mlp", "svm", "hgb"] if a.auto_pool else {"mlp": ["mlp"], "logreg": ["logreg"], "both": ["logreg", "mlp"]}[a.feat]
yv, yt = split["validation"][1], split["test"][1]
P = {}
for n in names:
    m = make(n).fit(Ftr, ytr)
    P[n] = {"validation": m.predict_proba(Fva), "test": m.predict_proba(Fte)}
    print(f"feature model {n:7s} validation acc {accuracy_score(yv, P[n]['validation'].argmax(1)):.4f}", flush=True)

pools = ({"logreg+mlp": ["logreg", "mlp"], "logreg+mlp+svm": ["logreg", "mlp", "svm"],
          "logreg+mlp+hgb": ["logreg", "mlp", "hgb"], "all4": ["logreg", "mlp", "svm", "hgb"]}
         if a.auto_pool else {"+".join(names): names})
pool_p = lambda pn, s: np.mean([P[n][s] for n in pools[pn]], axis=0)

print("\nw = weight on SongNet (1-w on feature models)")
best_pool, best_w, best_v, ncand = None, 1.0, -1, 0
for pn in pools:
    for w in ([] if a.w is not None else np.linspace(0, 1, 11)):
        v = accuracy_score(yv, (w * S["validation"] + (1 - w) * pool_p(pn, "validation")).argmax(1)); ncand += 1
        print(f"  pool={pn:15s} w={w:.1f}  validation acc {v:.4f}")
        if v > best_v: best_pool, best_w, best_v = pn, w, v
if a.w is not None:
    best_pool, best_w = list(pools)[0], a.w; print(f"using fixed w = {best_w:.2f}")
else:
    print(f"\ncandidates evaluated on validation: {ncand}")
    print(f"chosen pool = {best_pool}, w = {best_w:.1f} (validation {best_v:.4f})")
Fp = {"test": pool_p(best_pool, "test")}
print(f"SongNet only   test acc: {accuracy_score(yt, S['test'].argmax(1)):.4f}")
print(f"Features only  test acc: {accuracy_score(yt, Fp['test'].argmax(1)):.4f}")
final = (best_w * S["test"] + (1 - best_w) * Fp["test"]).argmax(1)
print(f"HYBRID         test acc: {accuracy_score(yt, final):.4f}\n")
print(classification_report(yt, final, target_names=classes, digits=3))
