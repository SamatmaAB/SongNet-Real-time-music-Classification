# SongNet: Real-time Music Classification (reproduction + extensions)

Reproduces Stanford CS229 (2018) "SongNet" on FMA-small (8 genres) and compares it with classical baselines.

## Setup
```bash
pip install -r requirements.txt
```
Download `fma_small.zip` and `fma_metadata.zip` from https://github.com/mdeff/fma and unzip into `data/`
(so you have `data/fma_small/` and `data/fma_metadata/`).

## Run
```bash
python src/data.py --meta data/fma_metadata --audio data/fma_small --out data/processed
python src/baselines.py --meta data/fma_metadata
python src/train.py --head dense     # paper-style model
python src/train.py --head gru       # extension: causal GRU head
```
Outputs (checkpoints, test report, confusion matrix) go to `results/`.

## Notes
- Uses FMA's official split (80/10/10), not the paper's 70/20/10.
- Baselines use FMA's precomputed features (`features.csv`) without metadata.
## Demo
```bash
streamlit run app.py
```
Needs a trained `results/songnet_<head>.pt` (run training first). Upload a song, press play, and watch the genre guess update every ~0.37 s.
