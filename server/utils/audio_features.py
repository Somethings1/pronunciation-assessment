import numpy as np
import librosa
from g2p_en import G2p
import syllapy
import re

# Khởi tạo G2P
g2p = G2p()


def get_syllables_and_phonemes(word):
    """
    Phân tích từ thành danh sách âm tiết và phoneme tương ứng.
    Ví dụ: "banana" -> [('B', 'AH0'), ('N', 'AE1'), ('N', 'AH0')] (Minh họa)
    """
    # 1. Lấy Phonemes kèm Stress markers (0, 1, 2) từ CMU dict
    phonemes = g2p(word)
    phonemes = [p for p in phonemes if p not in [' ', "'", ',']]

    # 2. Chia âm tiết (Heuristic dựa trên thư viện syllapy + matching phoneme)
    syllable_count = syllapy.count(word)

    # Gom nhóm phoneme vào syllable (Logic đơn giản hóa: Gom quanh nguyên âm)
    syllables_data = []
    current_syl = []

    for p in phonemes:
        current_syl.append(p)
        if any(char.isdigit() for char in p):
            syllables_data.append(current_syl)
            current_syl = []

    # Gán nốt phần dư vào âm tiết cuối
    if current_syl and syllables_data:
        syllables_data[-1].extend(current_syl)
    elif current_syl:
        syllables_data.append(current_syl)

    # Fallback nếu chia sai số lượng
    if len(syllables_data) != syllable_count:
        syllable_count = len(syllables_data)

    return syllables_data


def extract_acoustic_features_19(y_syl, sr):
    """
    Trích xuất 19 features âm thanh theo
    (Intensity, Duration, Pitch stats + Sonority contour approximation)
    """
    if len(y_syl) < 512:
        return np.zeros(19)

    # 1. Duration (1)
    duration = len(y_syl) / sr

    # 2. Pitch (F0)
    f0, _, _ = librosa.pyin(y_syl, fmin=50, fmax=300, sr=sr, frame_length=1024)
    f0 = f0[~np.isnan(f0)]

    if len(f0) == 0:
        pitch_stats = [0.0] * 6
    else:
        # Mean, Std, Max, Min, Range, Slope (6)
        pitch_stats = [
            np.mean(f0), np.std(f0), np.max(f0), np.min(f0), np.ptp(f0),
            np.polyfit(np.arange(len(f0)), f0, 1)[0] if len(f0) > 1 else 0
        ]

    # 3. Intensity / Energy (RMS)
    rms = librosa.feature.rms(y=y_syl)[0]
    # Mean, Std, Max, Min, Range, Slope (6)
    rms_stats = [
        np.mean(rms), np.std(rms), np.max(rms), np.min(rms), np.ptp(rms),
        np.polyfit(np.arange(len(rms)), rms, 1)[0] if len(rms) > 1 else 0
    ]

    # 4. Sonority / Spectral Info (6 features còn lại để đủ 19)
    spec_flat = librosa.feature.spectral_flatness(y=y_syl)[0]
    spec_roll = librosa.feature.spectral_rolloff(y=y_syl, sr=sr)[0]

    sonority_stats = [
        np.mean(spec_flat), np.max(spec_flat), np.min(spec_flat),
        np.mean(spec_roll), np.max(spec_roll), np.min(spec_roll)
    ]

    # Tổng: 1 + 6 + 6 + 6 = 19 features
    feats = [duration] + pitch_stats + rms_stats + sonority_stats
    return np.array(feats[:19])


def extract_context_features_19(syllable_data, index, total_syls):
    """
    Trích xuất 19 features ngữ cảnh theo
    (Nucleus type, neighboring phoneme category, word position...)
    Lưu ý: Đây là ONE-HOT ENCODING simulation.
    """
    feats = []

    # Phonemes trong syllable hiện tại (ví dụ: ['B', 'AA1', 'N'])
    phonemes = syllable_data

    # --- Group 1: Vị trí trong từ (3 features) ---
    if index == 0:
        feats.extend([1, 0, 0])
    elif index == total_syls - 1:
        feats.extend([0, 0, 1])
    else:
        feats.extend([0, 1, 0])

    # --- Group 2: Nucleus Type (Vowel info) (5 features) ---
    # Tìm nguyên âm trong âm tiết
    nucleus = next((p for p in phonemes if any(char.isdigit() for char in p)), "UNK")
    nucleus_clean = re.sub(r'\d+', '', nucleus)

    # Is Monophthong? (AA, AE, IH...) vs Diphthong (AY, EY, OW...)?
    is_diphthong = nucleus_clean in ['AY', 'EY', 'OY', 'AW', 'OW']
    feats.append(1 if is_diphthong else 0)

    # Vowel Height/Backness (Simplified 4 bits)
    feats.extend([0, 1, 0, 1])

    # --- Group 3: Structure (Onset/Coda) (5 features) ---
    has_onset = 1 if not any(char.isdigit() for char in phonemes[0]) else 0
    has_coda = 1 if not any(char.isdigit() for char in phonemes[-1]) else 0
    num_phonemes = len(phonemes)
    feats.extend([has_onset, has_coda, num_phonemes, 0, 0])

    # --- Group 4: Neighbor Context (6 features) ---
    # Preceding syllable existed? Following syllable existed?
    has_prev = 1 if index > 0 else 0
    has_next = 1 if index < total_syls - 1 else 0
    feats.extend([has_prev, has_next, 0, 0, 0, 0])

    # Resize cho đúng 19
    return np.array(feats[:19])


def extract_full_features(y, sr, word):
    """
    Hàm Main gọi từ Service.
    Trả về: (1, num_syllables, 38)
    """
    # 1. Phân tích Text
    syl_structure = get_syllables_and_phonemes(word)
    num_syl = len(syl_structure)
    if num_syl == 0:
        return np.zeros((1, 1, 38))

    # 2. Phân tích Audio (Segmentation)
    # Chia đều audio cho số âm tiết (Vì không có Force Alignment xịn)
    # [cite: 107] Paper dùng Syllable Segmentation từ alignment.
    # Ta dùng mẹo: chia theo năng lượng thấp (Energy Valleys) hoặc chia đều.
    # Chia đều cho nhanh (Fallback):
    syl_len_samples = len(y) // num_syl

    final_sequence = []

    for i in range(num_syl):
        # Cut audio
        start = i * syl_len_samples
        end = (i+1) * syl_len_samples if i < num_syl-1 else len(y)
        y_syl = y[start:end]

        # A. Acoustic (19)
        ac_feats = extract_acoustic_features_19(y_syl, sr)

        # B. Context (19)
        ctx_feats = extract_context_features_19(syl_structure[i], i, num_syl)

        # Combine -> 38 dims
        combined = np.concatenate([ac_feats, ctx_feats])
        final_sequence.append(combined)

    return np.array([final_sequence])
