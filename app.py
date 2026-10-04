"""SongNet real-time demo.  Run:  streamlit run app.py
Upload a song -> SongNet outputs genre probabilities for every ~0.37 s step.
The running average of those steps is the model's live guess as the song plays.
"""
import sys, json, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd, torch, librosa, streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from data import audio_to_mel, SR, HOP
from model import SongNet

CLASSES = json.load(open(ROOT / "data/processed/classes.json"))
STEP_SEC = 8 * HOP / SR          # 3 max-pools of 2 -> one output step per 8 mel frames

st.set_page_config(page_title="SongNet", page_icon="🎵")
st.title("🎵 SongNet: Real-time Genre Classification")

head = st.sidebar.radio("Model head", ["dense", "gru"], help="dense = paper-style, gru = our extension")
max_sec = st.sidebar.slider("Max seconds to analyze", 30, 120, 60, 10)


@st.cache_resource
def load_model(head):
    ck = ROOT / "results" / f"songnet_{head}.pt"
    nm = ROOT / "results" / f"norm_{head}.json"
    if not ck.exists():
        return None, None
    m = SongNet(n_classes=len(CLASSES), head=head)
    m.load_state_dict(torch.load(ck, map_location="cpu")); m.eval()
    return m, json.load(open(nm))


@st.cache_data(show_spinner="Analyzing audio...")
def analyze(data: bytes, suffix: str, head: str, max_sec: int):
    model, norm = load_model(head)
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = Path(tmp_dir) / f"upload{suffix}"
        audio_path.write_bytes(data)
        y, _ = librosa.load(audio_path, sr=SR, duration=max_sec)
    mel = ((audio_to_mel(y) - np.reshape(norm["mean"], (-1, 1))) / np.reshape(norm["std"], (-1, 1))).astype(np.float32)
    with torch.no_grad():
        p = torch.softmax(model(torch.from_numpy(mel)[None]), -1)[0].numpy()  # (T', C)
    return p


model, _ = load_model(head)
if model is None:
    st.error(f"No trained model found: results/songnet_{head}.pt. Run `python src/train.py --head {head}` first.")
    st.stop()

up = st.file_uploader("Upload a song (mp3 / wav)", type=["mp3", "wav", "ogg", "flac"])
if not up:
    st.info("Upload a clip to begin.")
    st.stop()

data = up.getvalue()
probs = analyze(data, Path(up.name).suffix or ".mp3", head, max_sec)
T = len(probs)
running = np.cumsum(probs, 0) / np.arange(1, T + 1)[:, None]   # live song-level guess at each step

st.subheader("Live prediction")
c1, c2 = st.columns([1, 2])
audio_ph = st.empty()
audio_ph.audio(data)
label_ph, bar_ph = st.empty(), st.empty()


def show(i):
    top = int(running[i].argmax())
    label_ph.markdown(f"**t = {i * STEP_SEC:5.1f}s → {CLASSES[top]}** ({running[i][top]:.0%})")
    bar_ph.bar_chart(pd.Series(running[i], index=CLASSES))


if st.button("▶ Play with live classification"):
    audio_ph.audio(data, autoplay=True)
    for i in range(T):
        show(i); time.sleep(STEP_SEC)
else:
    show(T - 1)

st.subheader("Scrub through time")
i = st.slider("Time (s)", 0.0, (T - 1) * STEP_SEC, (T - 1) * STEP_SEC, STEP_SEC)
idx = int(round(i / STEP_SEC))
st.write(f"Running guess at {i:.1f}s: **{CLASSES[int(running[idx].argmax())]}**")

st.subheader("Genre probabilities over time")
st.line_chart(pd.DataFrame(running, columns=CLASSES, index=np.round(np.arange(T) * STEP_SEC, 1)))
st.caption(f"Final prediction: **{CLASSES[int(running[-1].argmax())]}** (mean of {T} per-step predictions)")
