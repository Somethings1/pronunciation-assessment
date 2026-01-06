import sys
import os
import glob
import numpy as np
import librosa
import tgt
from tqdm import tqdm
from g2p_en import G2p
import re
import pickle
import zipfile
import shutil
import nltk
import warnings

warnings.filterwarnings('ignore')

# --- IMPORT (Giữ nguyên logic cũ của ông) ---
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    try:
        from server.utils.audio_features import extract_full_features
    except ImportError:
        from app.utils.audio_features import extract_full_features
except ImportError:
    print("❌ Lỗi Import audio_features! Check lại sys.path.")
    exit()

# --- CẤU HÌNH ---
# SỬA LẠI ĐƯỜNG DẪN CỦA ÔNG Ở ĐÂY
DATASET_ROOT = "L2-ARCTIC"

OUTPUT_DIR = "data_train_ready"
TEMP_EXTRACT_DIR = "temp_unzip_zone"
TEMP_CHUNK_DIR = "temp_chunks"
MAX_SEQ_LEN = 10
CHUNK_SIZE = 1000

nltk.download('averaged_perceptron_tagger_eng')

g2p = G2p()

def get_stress_labels(word_text, num_syllables_audio):
    # (Logic cũ)
    phonemes = g2p(word_text)
    stress_phonemes = [p for p in phonemes if any(char.isdigit() for char in p)]
    labels = []
    for p in stress_phonemes:
        stress_val = int(re.search(r'\d+', p).group())
        labels.append(1 if stress_val == 1 else 0)
    if len(labels) != num_syllables_audio:
        return None
    return labels

def find_textgrid_smart(wav_path):
    """
    Tìm file TextGrid tương ứng với wav_path một cách thông minh hơn.
    Bất chấp wav nằm trong folder 'wav', 'Wav', 'WAV'...
    Và TextGrid nằm trong 'textgrid', 'TextGrid'...
    """
    wav_dir = os.path.dirname(wav_path) # .../YBAA/wav
    speaker_dir = os.path.dirname(wav_dir) # .../YBAA
    filename_no_ext = os.path.splitext(os.path.basename(wav_path))[0] # YBAA_001

    # 1. Quét folder cha xem folder TextGrid tên là gì
    if not os.path.exists(speaker_dir):
        return None

    subdirs = os.listdir(speaker_dir)
    textgrid_dir_name = None

    # Tìm folder nào có chữ "textgrid" (không phân biệt hoa thường)
    for d in subdirs:
        if d.lower() == "textgrid":
            textgrid_dir_name = d
            break

    if not textgrid_dir_name:
        return None # Không thấy folder textgrid đâu

    textgrid_dir_path = os.path.join(speaker_dir, textgrid_dir_name)

    # 2. Tìm file TextGrid trong folder đó (lại check hoa thường tiếp)
    # Thử các đuôi phổ biến
    candidates = [
        f"{filename_no_ext}.TextGrid",
        f"{filename_no_ext}.textgrid",
        f"{filename_no_ext}.TEXTGRID"
    ]

    for c in candidates:
        full_path = os.path.join(textgrid_dir_path, c)
        if os.path.exists(full_path):
            return full_path

    return None

def process_one_folder(folder_path, X_buffer, Y_buffer, debug_first_file=False):
    wav_files = glob.glob(os.path.join(folder_path, "**/*.wav"), recursive=True)

    if debug_first_file:
        print(f"\n🔍 DEBUG: Tìm thấy {len(wav_files)} file wav trong {folder_path}")
        if len(wav_files) > 0:
            print(f"   -> File đầu tiên: {wav_files[0]}")
            tg_test = find_textgrid_smart(wav_files[0])
            if tg_test:
                print(f"   -> ✅ Tìm thấy TextGrid: {tg_test}")
            else:
                print(f"   -> ❌ KHÔNG tìm thấy TextGrid cho file này. Cấu trúc folder có vấn đề.")
                # In ra cấu trúc folder cha để soi
                wav_dir = os.path.dirname(wav_files[0])
                parent = os.path.dirname(wav_dir)
                print(f"   -> Nội dung folder '{os.path.basename(parent)}': {os.listdir(parent)}")

    for wav_path in wav_files:
        # Dùng hàm tìm thông minh
        tg_path = find_textgrid_smart(wav_path)

        if not tg_path:
            continue

        try:
            tg = tgt.read_textgrid(tg_path)
            y, sr = librosa.load(wav_path, sr=16000)
            words_tier = tg.get_tier_by_name("words")

            for interval in words_tier:
                word = interval.text.strip().upper()
                if word in ["", "SP", "SIL"] or len(word) < 2: continue

                start_sample = int(interval.start_time * sr)
                end_sample = int(interval.end_time * sr)
                y_word = y[start_sample:end_sample]
                if len(y_word) < 1000: continue

                features = extract_full_features(y_word, sr, word)
                if features.shape[1] == 0: continue

                labels = get_stress_labels(word, features.shape[1])
                if labels is None: continue

                # Pad/Cut
                feat_mat = features[0]
                if len(feat_mat) > MAX_SEQ_LEN:
                    feat_mat = feat_mat[:MAX_SEQ_LEN]
                    labels = labels[:MAX_SEQ_LEN]
                else:
                    pad_len = MAX_SEQ_LEN - len(feat_mat)
                    feat_mat = np.pad(feat_mat, ((0, pad_len), (0, 0)), mode='constant')
                    labels = labels + [0] * pad_len

                X_buffer.append(feat_mat)
                Y_buffer.append(labels)

        except Exception as e:
            # Nếu debug thì in lỗi ra xem
            if debug_first_file:
                print(f"   -> ⚠ Lỗi xử lý audio/feature: {e}")
            continue

    return X_buffer, Y_buffer

def save_chunk(X, Y):
    if not X: return
    os.makedirs(TEMP_CHUNK_DIR, exist_ok=True)
    chunk_id = len(glob.glob(os.path.join(TEMP_CHUNK_DIR, "chunk_*.pkl")))
    with open(os.path.join(TEMP_CHUNK_DIR, f"chunk_{chunk_id}.pkl"), "wb") as f:
        pickle.dump((X, Y), f)
    print(f" -> Saved chunk {chunk_id} ({len(X)} samples)")

def main():
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ Sai đường dẫn dataset: {DATASET_ROOT}")
        return

    # --- SỬA 1: KHÔNG ĐƯỢC XÓA FOLDER CHUNK NỮA ---
    # Dọn folder giải nén tạm thì được, nhưng cấm dọn folder chunk
    if os.path.exists(TEMP_EXTRACT_DIR): shutil.rmtree(TEMP_EXTRACT_DIR)
    # if os.path.exists(TEMP_CHUNK_DIR): shutil.rmtree(TEMP_CHUNK_DIR) <--- XÓA HOẶC COMMENT DÒNG NÀY NGAY

    zip_files = sorted(glob.glob(os.path.join(DATASET_ROOT, "*.zip"))) # Sort để thứ tự ổn định
    print(f"📦 Tìm thấy {len(zip_files)} file zip.")

    # --- SỬA 2: CHECK LOG XEM ĐÃ LÀM ĐẾN ĐÂU ---
    processed_log_path = "processed_zips.txt"
    processed_zips = set()
    if os.path.exists(processed_log_path):
        with open(processed_log_path, "r") as f:
            processed_zips = set(line.strip() for line in f)
        print(f"♻️ Đã xử lý xong {len(processed_zips)} file zip trước đó. Skipping...")

    X_buffer = []
    Y_buffer = []

    # Duyệt file zip
    for i, zip_path in enumerate(tqdm(zip_files, desc="Processing Zips")):
        zip_name = os.path.basename(zip_path)

        # Nếu file này đã làm rồi thì Next luôn
        if zip_name in processed_zips:
            continue

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(TEMP_EXTRACT_DIR)
        except zipfile.BadZipFile:
            print(f"⚠️ Lỗi file zip: {zip_name}")
            continue

        # Xử lý
        X_buffer, Y_buffer = process_one_folder(TEMP_EXTRACT_DIR, X_buffer, Y_buffer, debug_first_file=False)

        # Lưu chunk nếu đầy
        if len(X_buffer) >= CHUNK_SIZE:
            save_chunk(X_buffer, Y_buffer)
            X_buffer = []
            Y_buffer = []

        shutil.rmtree(TEMP_EXTRACT_DIR)

        # --- SỬA 3: GHI VÀO SỔ NAM TÀO ---
        with open(processed_log_path, "a") as f:
            f.write(zip_name + "\n")

    # Save nốt phần dư
    if X_buffer:
        save_chunk(X_buffer, Y_buffer)

    # Merge (Chỉ merge khi đã chạy xong hết hoặc muốn chốt sổ)
    # Mẹo: Nếu muốn merge giữa chừng để train thử thì chạy code merge riêng.
    # Còn code này cứ để nó chạy hết list zip đã.

    print("\n🏗 Merging data...")
    all_X = []
    all_Y = []
    chunks = glob.glob(os.path.join(TEMP_CHUNK_DIR, "*.pkl"))

    if not chunks:
        print("❌ Chưa có chunk nào.")
        return

    for pkl in tqdm(chunks, desc="Merging"):
        with open(pkl, "rb") as f:
            x, y = pickle.load(f)
            all_X.extend(x)
            all_Y.extend(y)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X_train = np.array(all_X)
    Y_train = np.array(all_Y)[..., np.newaxis]

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "Y_train.npy"), Y_train)

    print(f"\n🎉 SUCCESS! Tổng số mẫu: {len(X_train)}")
    print(f"📁 File output nằm tại: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
