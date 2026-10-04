import torch, torch.nn as nn, torch.nn.functional as F


class SongNet(nn.Module):
    """1D-conv over time (mel bins = channels) -> per-timestep genre logits.
    head='dense': time-distributed linear layer (as in the paper)
    head='gru'  : causal GRU before the linear layer (real recurrent head)
    Song-level prediction = mean of per-timestep probabilities.
    """
    def __init__(self, n_mels=128, n_classes=8, channels=(128, 256, 256), k=5, p=0.25, head="dense"):
        super().__init__()
        layers, c = [], n_mels
        for o in channels:
            layers += [nn.Conv1d(c, o, k, padding=k // 2), nn.BatchNorm1d(o),
                       nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(p)]
            c = o
        self.cnn = nn.Sequential(*layers)
        self.head = head
        if head == "gru":
            self.rnn = nn.GRU(c, 256, batch_first=True)  # unidirectional -> usable in real time
            c = 256
        self.fc = nn.Linear(c, n_classes)

    def forward(self, x):                 # x: (B, n_mels, T)
        h = self.cnn(x).transpose(1, 2)   # (B, T', C)
        if self.head == "gru":
            h, _ = self.rnn(h)
        return self.fc(h)                 # (B, T', n_classes)


def song_log_probs(logits):
    """Mean of per-timestep softmax -> log-probabilities per song."""
    return torch.log(F.softmax(logits, -1).mean(1) + 1e-8)
