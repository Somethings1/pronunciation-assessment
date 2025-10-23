import os
from pathlib import Path
import pandas as pd
import soundfile as sf
import io
import subprocess

def prepare_speechocean762(output_base: str = "data"):
    base = Path(output_base)
    input_base = base / "mfa_input"
    input_base.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rm', '-rf', input_base])

    # HF parquet paths
    splits = {
        "train": "data/train-00000-of-00001.parquet",
        "test": "data/test-00000-of-00001.parquet",
    }

    all_records = []

    for split, rel_path in splits.items():
        parquet_url = f"hf://datasets/mispeech/speechocean762/{rel_path}"
        print(f"Loading {split} split from {parquet_url} ...")
        df = pd.read_parquet(parquet_url)
        print(f"{split}: {len(df)} rows")

        for i, row in df.iterrows():
            audio_info = row["audio"]
            speaker = str(row.get("speaker", "unknown")).strip()
            if not speaker:
                speaker = "unknown"

            speaker_dir = input_base / split / speaker
            speaker_dir.mkdir(parents=True, exist_ok=True)

            # Extract audio bytes + create example ID
            if isinstance(audio_info, dict):
                ex_id = Path(audio_info.get("path", f"{i}")).stem
                audio_bytes = audio_info.get("bytes")
            else:
                ex_id = str(i)
                audio_bytes = audio_info

            # Write WAV file
            wav_path = speaker_dir / f"{ex_id}.wav"
            with io.BytesIO(audio_bytes) as bio:
                data, sr = sf.read(bio)
                sf.write(wav_path, data, sr)

            # Write transcript (.lab)
            text = str(row.get("text", "")).strip()
            lab_path = speaker_dir / f"{ex_id}.lab"
            with open(lab_path, "w", encoding="utf-8") as f:
                f.write(text + "\n")

            # Record metadata
            rec = row.to_dict()
            rec["split"] = split
            rec["id"] = ex_id
            rec["speaker"] = speaker
            all_records.append(rec)

    # Combine metadata
    meta_df = pd.DataFrame(all_records)

    # Drop heavy/redundant columns
    drop_cols = ["audio", "text"]
    meta_df = meta_df.drop(columns=[c for c in drop_cols if c in meta_df.columns], errors="ignore")

    # Write assessment CSV
    csv_path = base / "assessment.csv"
    meta_df.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"\n✅ Done. Output structure:")
    print(f"- {input_base}/train/<speaker_id>/*.wav + *.lab")
    print(f"- {input_base}/test/<speaker_id>/*.wav + *.lab")
    print(f"- {csv_path}")

if __name__ == "__main__":
    prepare_speechocean762("data")
