import os
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import io
import syllapy
from g2p_en import G2p

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.append(project_root)
# Hack path để import module của chúng ta
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from server.services.stress_service import StressEvaluator

# --- CẤU HÌNH ---
MODEL_PATH = "./checkpoints/stress_model.h5"
SCALER_PATH = "./checkpoints/scaler_params.json"
SAMPLE_RATE = 16000
DURATION = 2.5 # Thu âm 2.5 giây mỗi từ (đủ cho từ dài)

# Danh sách từ để test
TEST_WORDS = [
    "banana",           # Cơ bản (ba-NA-na) -> Stress index 1
    "education",        # Trung bình (e-du-CA-tion) -> Stress index 2
    "computer",         # (com-PU-ter) -> Stress index 1
    "university",       # (u-ni-VER-si-ty) -> Stress index 2
    "agriculturalization", # SIÊU DÀI (a-gri-cul-tu-ra-li-ZA-tion) -> Stress index 6
    "present",          # Từ đa nghĩa (Noun: PRE-sent / Verb: pre-SENT). Thử nói kiểu Noun xem.
]

def record_audio(duration, sr=16000):
    print("🎙  Đang thu âm...", end="\r")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Chờ thu xong
    print("✅ Thu xong!        ")
    return recording.flatten()

def numpy_to_wav_bytes(audio_data, sr):
    """Chuyển numpy array sang wav bytes để giả lập file upload"""
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sr, format='WAV')
    buffer.seek(0)
    return buffer.read()

def main():
    # 1. Load Model
    print("⏳ Đang khởi động hệ thống...")
    if not os.path.exists(MODEL_PATH):
        print("❌ Chưa thấy file model. Train xong chưa đại ca?")
        return

    evaluator = StressEvaluator(MODEL_PATH, SCALER_PATH)
    g2p = G2p()

    print("\n" + "="*50)
    print("   TEST TRỌNG ÂM TRỰC TIẾP TỪ MICROPHNONE")
    print("="*50)
    print(f"Danh sách từ test: {', '.join(TEST_WORDS)}")
    print("Nhấn Enter để bắt đầu từng từ...")

    for word in TEST_WORDS:
        input(f"\n👉 Nhấn ENTER để nói từ: '{word.upper()}'")

        # Đếm ngược cho chuyên nghiệp
        for i in range(3, 0, -1):
            print(f" {i}...", end="\r", flush=True)
            time.sleep(0.5)

        # Thu âm
        audio_data = record_audio(DURATION, SAMPLE_RATE)

        # Convert sang bytes (giả lập như lúc upload qua API)
        audio_bytes = numpy_to_wav_bytes(audio_data, SAMPLE_RATE)

        # Dự đoán
        result = evaluator.predict(audio_bytes, word)

        if "error" in result:
            print(f"❌ Lỗi: {result['error']}")
            continue

        # In kết quả đẹp
        # Lấy phoneme để đối chiếu (Ground Truth tham khảo)
        phonemes = g2p(word)
        phoneme_str = " ".join(phonemes)

        # Dự đoán của model
        pred_idx = result['stress_index']
        prob = result['stress_probability']
        scores = result['raw_scores']

        # Tạo visual thanh điểm
        # Ví dụ: [0.1, 0.9, 0.05] ->  _  █  _
        visual_bar = ""
        for s in scores:
            if s > 0.5: visual_bar += "🔴 " # Trọng âm
            else: visual_bar += "⚪ "       # Không trọng âm

        print(f"🗣  Bạn nói:     {word}")
        print(f"🤖 Phonemes:    {phoneme_str}")
        print(f"🎯 Kết quả:     Trọng âm rơi vào âm tiết thứ {pred_idx + 1} (Index {pred_idx})")
        print(f"📊 Confidence:  {prob:.4f} (Độ tự tin)")
        print(f"🎼 Visual:      {visual_bar}")
        print("-" * 30)

    print("\n🎉 Hoàn thành bài test!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🚫 Đã dừng chương trình.")
    except Exception as e:
        print(f"\n❌ Lỗi không mong muốn: {e}")
        print("💡 Gợi ý: Nếu lỗi liên quan đến PortAudio/SoundDevice, hãy cài 'brew install portaudio'")
