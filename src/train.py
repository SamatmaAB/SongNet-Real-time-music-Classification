"""Train + evaluate SongNet (v2: random time crops, SpecAugment-style masking,
per-mel-bin normalization, AdamW).
Usage: python src/train.py --data data/processed --head dense
"""
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from model import SongNet, song_log_probs

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed")
ap.add_argument("--head", choices=["dense", "gru"], default="dense")
ap.add_argument("--epochs", type=int, default=80)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=1e-2)
ap.add_argument("--crop", type=int, default=320, help="training crop length in frames (640 = full clip)")
ap.add_argument("--patience", type=int, default=15)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results")
ap.add_argument("--trainval", action="store_true", help="train on train+validation for a FIXED number of epochs (no model selection); test stays held out")
a = ap.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
d = Path(a.data); out = Path(a.out); out.mkdir(exist_ok=True)
classes = json.load(open(d / "classes.json"))
L = lambda s: (torch.from_numpy(np.load(d / f"{s}_X.npy")), torch.from_numpy(np.load(d / f"{s}_y.npy")))
(Xtr, ytr), (Xva, yva), (Xte, yte) = L("training"), L("validation"), L("test")

if a.trainval:
    Xtr = torch.cat([Xtr, Xva]); ytr = torch.cat([ytr, yva])

sample = Xtr[:1000].float()
mean = sample.mean(dim=(0, 2), keepdim=True)[0]            # (n_mels, 1) per-bin stats
std = sample.std(dim=(0, 2), keepdim=True)[0] + 1e-5
json.dump({"mean": mean.flatten().tolist(), "std": std.flatten().tolist()}, open(out / f"norm_{a.head}.json", "w"))
mean, std = mean.to(dev), std.to(dev)
prep = lambda x: ((x.to(dev).float() - mean) / std)


def augment(X, idx):
    """Random time crop per sample + one freq mask and one time mask."""
    T, crop = X.shape[2], min(a.crop, X.shape[2])
    starts = np.random.randint(0, T - crop + 1, size=len(idx))
    x = prep(torch.stack([X[j, :, s:s + crop] for j, s in zip(idx.tolist(), starts)]))
    for b in range(len(x)):
        f = np.random.randint(0, 16); f0 = np.random.randint(0, 128 - f + 1)
        t = np.random.randint(0, 24); t0 = np.random.randint(0, crop - t + 1)
        x[b, f0:f0 + f, :] = 0; x[b, :, t0:t0 + t] = 0
    return x


model = SongNet(n_classes=len(classes), head=a.head).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min" if a.trainval else "max", factor=0.5, patience=4)


@torch.no_grad()
def predict(X):
    model.eval(); preds = []
    for i in range(0, len(X), 64):
        preds.append(song_log_probs(model(prep(X[i:i + 64]))).argmax(1).cpu())
    return torch.cat(preds).numpy()


best, bad = 0, 0
for ep in range(a.epochs):
    model.train(); perm = torch.randperm(len(Xtr)); tot = 0
    for i in range(0, len(perm), a.bs):
        idx = perm[i:i + a.bs]
        loss = F.nll_loss(song_log_probs(model(augment(Xtr, idx))), ytr[idx].to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
    if a.trainval:
        sched.step(tot / len(perm))
        print(f"epoch {ep+1:02d} loss {tot/len(perm):.4f} (train+val, fixed schedule)", flush=True)
        continue
    val = accuracy_score(yva.numpy(), predict(Xva)); sched.step(val)
    print(f"epoch {ep+1:02d} loss {tot/len(perm):.4f} val_acc {val:.4f}", flush=True)
    if val > best:
        best, bad = val, 0; torch.save(model.state_dict(), out / f"songnet_{a.head}.pt")
    else:
        bad += 1
        if bad >= a.patience: print("early stop"); break

if a.trainval:
    torch.save(model.state_dict(), out / f"songnet_{a.head}.pt")
model.load_state_dict(torch.load(out / f"songnet_{a.head}.pt", map_location=dev))
pred = predict(Xte); acc = accuracy_score(yte.numpy(), pred)
print(f"\nTEST accuracy ({a.head}): {acc:.4f}")
rep = classification_report(yte.numpy(), pred, target_names=classes, digits=3)
print(rep)
open(out / f"report_{a.head}.txt", "w").write(f"test_acc={acc:.4f}\n{rep}")
cm = confusion_matrix(yte.numpy(), pred)
plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(out / f"confusion_{a.head}.png", dpi=150)
