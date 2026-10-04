"""Evaluate saved checkpoints on the test set; averages probabilities if several runs are given.
Usage:
  python src/evaluate.py --data data/processed_random --runs results_random:dense
  python src/evaluate.py --data data/processed_random --runs results_random:dense results_random:gru
Each run is <results_dir>:<head>. Use --split validation to check validation accuracy instead.
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model import SongNet

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed")
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--split", default="test", choices=["test", "validation"])
ap.add_argument("--out", default="results_eval")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = Path(a.data)
classes = json.load(open(d / "classes.json"))
X = torch.from_numpy(np.load(d / f"{a.split}_X.npy")); y = np.load(d / f"{a.split}_y.npy")


@torch.no_grad()
def run_probs(res_dir, head):
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


all_p = []
for r in a.runs:
    res_dir, head = r.split(":")
    p = run_probs(res_dir, head); all_p.append(p)
    print(f"{r:35s} {a.split} accuracy: {accuracy_score(y, p.argmax(1)):.4f}")

final = np.mean(all_p, axis=0)
pred = final.argmax(1)
tag = "ensemble" if len(all_p) > 1 else "single"
print(f"\n{tag.upper()} {a.split} accuracy: {accuracy_score(y, pred):.4f}\n")
print(classification_report(y, pred, target_names=classes, digits=3))
Path(a.out).mkdir(exist_ok=True)
cm = confusion_matrix(y, pred)
plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(Path(a.out) / f"confusion_{tag}_{a.split}.png", dpi=150)
