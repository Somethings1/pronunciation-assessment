import pandas as pd
import numpy as np
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re

from transformers import AutoProcessor, Wav2Vec2ForCTC

# --- Configuration ---
GOP_ACOUSTIC_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
MIN_AUDIO_SAMPLES = 400 # A safe minimum length for the Wav2Vec2 model

# Input/output file paths
JOINED_CSV_PATH = Path("data/joined.csv")
ASSESSMENT_CSV_PATH = Path("data/assessment.csv")
AUDIO_BASE_PATH = Path("data/mfa_input/train")
OUTPUT_CSV_PATH = Path("data/gop.csv")
OUTPUT_PLOT_PATH = Path("correlation_comparison.png")

# --- 1. Setup Models ---
print(f"--> Loading model: {GOP_ACOUSTIC_MODEL}...")
processor = AutoProcessor.from_pretrained(GOP_ACOUSTIC_MODEL)
gop_model = Wav2Vec2ForCTC.from_pretrained(GOP_ACOUSTIC_MODEL)
print("--> Model loaded successfully.")

# --- 2. Load and Prepare DataFrames ---
print("--> Loading and merging CSV data...")
df_joined = pd.read_csv(JOINED_CSV_PATH, dtype={'file_name': object})
df_assessment = pd.read_csv(ASSESSMENT_CSV_PATH, dtype={'id': object, 'speaker': object})
df = pd.merge(df_joined, df_assessment, left_on="file_name", right_on="id", how="left")
df.dropna(subset=['speaker'], inplace=True)
print(f"--> Data prepared. Found {len(df)} phonemes to process.")

# --- 3. Calculate GOP Scores ---
gop_scores = []
cached_audio = {}
print("--> Calculating GOP scores for all phonemes...")
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    try:
        file_name, speaker_id = row['file_name'], row['speaker']
        start_time, end_time = row['begin'], row['end']
        phone_goodness_mfa = row['phone_goodness_mfa']

        if file_name not in cached_audio:
            wav_path = AUDIO_BASE_PATH / speaker_id / f"{file_name}.wav"
            audio_tensor, sample_rate = torchaudio.load(wav_path)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                audio_tensor = resampler(audio_tensor)
            cached_audio[file_name] = {"tensor": audio_tensor, "sr": 16000}
        
        audio_tensor, sample_rate = cached_audio[file_name]["tensor"], cached_audio[file_name]["sr"]

        start_sample, end_sample = int(start_time * sample_rate), int(end_time * sample_rate)
        
        # --- THIS IS THE FIX ---
        # Check if the audio segment is long enough for the model
        if (end_sample - start_sample) < MIN_AUDIO_SAMPLES:
            gop_scores.append(np.nan) # Append NaN and skip to the next phoneme
            continue
        
        segment = audio_tensor[:, start_sample:end_sample]

        with torch.no_grad():
            logits = gop_model(segment).logits
        
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        avg_log_probs = torch.mean(log_probs, dim=1).squeeze()
        log_prob_best_fit = torch.max(avg_log_probs).item()
        
        gop_score = phone_goodness_mfa - log_prob_best_fit
        gop_scores.append(gop_score)

    except Exception as e:
        # This will now only catch unexpected errors
        print(f"Unexpected error on row {index}: {e}")
        gop_scores.append(np.nan)

    if index > 500:
        break

# --- 4. Save and Plot ---
print("\n--> Saving results and plotting...")
df = df[:len(gop_scores)]
df['gop'] = gop_scores
df.to_csv(OUTPUT_CSV_PATH, index=False)

df_clean = df.dropna(subset=['phone_goodness_mfa', 'gop', 'phone_goodness_prof'])

if not df_clean.empty:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Comparison of Pronunciation Goodness Scores vs. Professional Ratings', fontsize=16)

    sns.regplot(ax=axes[0], x='phone_goodness_mfa', y='phone_goodness_prof', data=df_clean, scatter_kws={'alpha': 0.2}, line_kws={'color': 'red', 'linewidth': 3})
    axes[0].set_title('Original: MFA Log-Likelihood vs. Professional Score')
    axes[0].set_xlabel('MFA Raw Log-Likelihood')
    axes[0].set_ylabel('Professional Score')
    axes[0].grid(True)

    sns.regplot(ax=axes[1], x='gop', y='phone_goodness_prof', data=df_clean, scatter_kws={'alpha': 0.2}, line_kws={'color': 'green', 'linewidth': 3})
    axes[1].set_title('New: GOP Score vs. Professional Score')
    axes[1].set_xlabel('Calculated GOP Score')
    axes[1].set_ylabel('Professional Score')
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_PLOT_PATH)
    print(f"--> Done. Plot saved to {OUTPUT_PLOT_PATH}")
    plt.show()
else:
    print("--> No valid GOP scores were calculated, so no plot was generated.")