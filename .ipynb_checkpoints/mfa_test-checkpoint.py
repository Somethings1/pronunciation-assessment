import pandas as pd
import numpy as np
import soundfile as sf
import io
import requests
import textgrid
import torch
from speechbrain.inference.ASR import EncoderASR

# --- Configuration ---
MFA_SERVER_URL = "http://localhost:7749/align"
GOP_ACOUSTIC_MODEL = "speechbrain/asr-wav2vec2-commonvoice-en"

def calculate_gop_scores(textgrid_content: str, audio_data: np.ndarray, sample_rate: int, gop_model, phoneme_to_idx):
    """Calculates GOP scores using a pre-loaded SpeechBrain model."""
    print("--> 4. Calculating GOP Scores...")
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
    tg = textgrid.TextGrid.fromFile(io.StringIO(textgrid_content))
    phone_tier = tg.getFirst("phones")
    results = []

    for interval in phone_tier:
        if not interval.mark: continue
        start_time, end_time, phoneme_label = interval.minTime, interval.maxTime, interval.mark
        start_sample, end_sample = int(start_time * sample_rate), int(end_time * sample_rate)
        if start_sample >= end_sample: continue
        
        segment = audio_tensor[:, start_sample:end_sample]
        with torch.no_grad():
            log_probs = gop_model(segment)
        
        avg_log_probs = torch.mean(log_probs, dim=1).squeeze()
        clean_phoneme = phoneme_label.strip("012")
        if clean_phoneme not in phoneme_to_idx: continue

        log_prob_correct = avg_log_probs[phoneme_to_idx[clean_phoneme]]
        log_prob_best = torch.max(avg_log_probs)
        gop_score = (log_prob_correct - log_prob_best).item()
        results.append({"phoneme": phoneme_label, "gop_score": round(gop_score, 2)})
    return results

def main():
    # 1. Load the Speechocean dataset example using Pandas
    print("--> 1. Loading 'mispeech/speechocean762' example...")
    parquet_url = "hf://datasets/mispeech/speechocean762/data/train-00000-of-00001.parquet"
    df = pd.read_parquet(parquet_url)
    first_example = df.iloc[0]
    audio_info = first_example["audio"]
    raw_audio_bytes = audio_info["bytes"]
    audio_data, sample_rate = sf.read(io.BytesIO(raw_audio_bytes))
    transcript = first_example["text"]
    print(f"    - Loaded transcript: '{transcript}'")

    # 2. Call the MFA Server to get the alignment
    print(f"--> 2. Sending audio to MFA server at {MFA_SERVER_URL}...")
    try:
        files = {'audio': ('audio.wav', io.BytesIO(raw_audio_bytes), 'audio/wav')}
        data = {'text': transcript}
        response = requests.post(MFA_SERVER_URL, files=files, data=data)
        response.raise_for_status() # Will raise an exception for bad status codes
        mfa_result = response.json()
        if mfa_result.get("status") != "success":
            print(f"Error from MFA server: {mfa_result.get('error')}")
            return
        textgrid_content = mfa_result["textgrid"]
        print("--> 3. Received TextGrid alignment successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not connect to MFA server. Is it running? Details: {e}")
        return

    # 3. Load the GOP model and calculate scores
    print("--> Loading SpeechBrain GOP model...")
    gop_model = EncoderASR.from_hparams(source=GOP_ACOUSTIC_MODEL)
    phoneme_map = gop_model.hparams.label_encoder.get_decoding_dict()
    phoneme_to_idx = {v: k for k, v in phoneme_map.items()}
    
    gop_results = calculate_gop_scores(textgrid_content, audio_data, sample_rate, gop_model, phoneme_to_idx)
    
    # 4. Display final results
    print("\n--- FINAL PHONEME ASSESSMENT RESULTS ---")
    if gop_results:
        for result in gop_results:
            print(f"Phoneme: {result['phoneme']:<5} | GOP Score: {result['gop_score']:.2f}")
    else:
        print("No results to display.")

if __name__ == "__main__":
    main()