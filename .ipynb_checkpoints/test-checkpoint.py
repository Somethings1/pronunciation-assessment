import os, math, torch, librosa, numpy as np, pandas as pd
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from tqdm import tqdm

# ---------------- CONFIG ----------------
MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"
JOINED_CSV = "data/joined.csv"
ASSESSMENT_CSV = "data/assessment.csv"
AUDIO_ROOT = Path("data/mfa_input/train")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

# ---------------- MODEL ----------------
print(f"[INFO] Loading {MODEL_NAME} ...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device)
model.eval()

id2label = model.config.id2label
label2id = {v: int(k) for k, v in id2label.items()}
blank_id = model.config.pad_token_id if hasattr(model.config, "pad_token_id") else 0

print(f"[INFO] Model vocab size: {len(id2label)}")
print("Sample labels:", list(id2label.values())[:40])

# ---------------- DATA ----------------
joined = pd.read_csv(JOINED_CSV, dtype=str)
assessment = pd.read_csv(ASSESSMENT_CSV, dtype=str)
file_to_speaker = assessment.set_index("id")["speaker"].to_dict()
joined["speaker"] = joined["file_name"].map(file_to_speaker)

def get_start_end_cols(df):
    s, e = None, None
    for c in ["start","interval_start","begin"]:
        if c in df.columns: s=c
    for c in ["end","interval_end","end_time"]:
        if c in df.columns: e=c
    return s,e

start_col, end_col = get_start_end_cols(joined)
if not start_col or not end_col:
    raise ValueError("Start/end columns not found")

grouped = joined.groupby("file_name")

# ---------------- HELPERS ----------------
def time_to_frame_indices(start_s, end_s, audio_dur, num_frames):
    s = max(0, min(start_s, audio_dur))
    e = max(0, min(end_s, audio_dur))
    if e <= s: return None, None
    sf = int(math.floor((s/audio_dur)*num_frames))
    ef = int(math.ceil((e/audio_dur)*num_frames))
    sf = max(0, min(sf, num_frames-1))
    ef = max(1, min(ef, num_frames))
    if ef <= sf: ef = sf+1
    return sf, ef

def compute_gop_ctc_align(log_probs, target_id, blank_id, s, e):
    slice_lp = log_probs[s:e]
    if slice_lp.shape[0]==0: return np.nan
    argmax = np.argmax(slice_lp, axis=1)
    mask = (argmax != blank_id)
    if mask.sum()==0: return np.nan
    slice_lp = slice_lp[mask]
    mean_log = slice_lp.mean(axis=0)
    target_val = mean_log[target_id]
    others = np.delete(mean_log, target_id)
    max_other = np.max(others)
    return float(target_val - max_other), float(target_val), float(max_other)

# ---------------- MAIN (one utterance) ----------------
for file_name, group in grouped:
    speaker = group["speaker"].iloc[0]
    wav_path = AUDIO_ROOT / speaker / f"{file_name}.wav"
    if not wav_path.exists():
        print(f"[WARN] Missing {wav_path}")
        continue

    print(f"\n[INFO] Processing file: {wav_path}")
    target_sr = 16000
    wav, _ = librosa.load(str(wav_path), sr=target_sr)
    duration = len(wav)/target_sr

    inputs = processor(wav, sampling_rate=target_sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits[0].cpu()
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).numpy()
    num_frames = log_probs.shape[0]
    print(f"[INFO] Duration={duration:.2f}s, frames={num_frames}")

    # ---- Per-phone loop ----
    for _, row in group.iterrows():
        start, end = float(row[start_col]), float(row[end_col])
        s, e = time_to_frame_indices(start, end, duration, num_frames)
        phone = row["phone"]
        phone_upper = phone.strip().upper()
        target_id = label2id.get(phone_upper)
        if target_id is None:
            print(f"  [MISSING] phone {phone} (no match in model labels)")
            continue
        gop, tval, oval = compute_gop_ctc_align(log_probs, target_id, blank_id, s, e)
        print(f"  {phone:>4s}: frames[{s}:{e}] GOP={gop:.3f} (target={tval:.3f}, other={oval:.3f})")

    print("\n[END] Stopped after first utterance for debug.\n")
    break
