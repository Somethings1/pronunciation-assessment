import uvicorn
import os
import io
import librosa
import numpy as np
import re
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from g2p_en import G2p
from server.utils.audio_processor import preprocess_audio

# Import Services
from server.services.stress_service import StressEvaluator
from server.services.gop_service import GOPEvaluator

app = FastAPI(
    title="Pronunciation Assessment API",
    description="API chấm điểm phát âm (GOP) và bắt lỗi trọng âm (Stress)",
    version="0.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIG PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRESS_MODEL_PATH = os.path.join(BASE_DIR, "models", "sylstress", "stress_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "sylstress", "scaler_params.json")
GOP_MODEL_PATH = os.path.join(BASE_DIR, "models", "ctcgop")

# Global instances
stress_service = None
gop_service = None
g2p = G2p()


@app.on_event("startup")
async def startup_event():
    global stress_service, gop_service
    print("🚀 Starting up API...")

    # Init Services
    if os.path.exists(STRESS_MODEL_PATH):
        stress_service = StressEvaluator(STRESS_MODEL_PATH, SCALER_PATH)
    else:
        print(f"⚠ Warning: Không tìm thấy Stress Model tại {STRESS_MODEL_PATH}")

    if os.path.exists(GOP_MODEL_PATH):
        gop_service = GOPEvaluator(GOP_MODEL_PATH)
    else:
        print(f"⚠ Warning: Không tìm thấy GOP Model tại {GOP_MODEL_PATH}")


def get_truth_stress(word):
    """Lấy trọng âm chuẩn từ từ điển CMU"""
    # G2P trả về: ['B', 'AH0', 'N', 'AE1', 'N', 'AH0']
    phonemes = g2p(word)
    # Lọc lấy số (stress marker)
    stress_seq = []
    for p in phonemes:
        if any(char.isdigit() for char in p):
            val = int(re.search(r'\d+', p).group())
            # Quy ước: 1 (Primary) -> 1, còn lại (0, 2) -> 0
            stress_seq.append(1 if val == 1 else 0)
    return stress_seq


@app.post("/assess")
async def assess_pronunciation(
    word: str = Form(..., description="Từ cần chấm điểm (vd: 'banana')"),
    audio: UploadFile = File(...)
):
    """
    Endpoint All-in-One:
    - Input: Audio + Word
    - Output: GOP Scores (Phoneme) + Stress Detection (Syllable)
    """
    if not stress_service or not gop_service:
        raise HTTPException(503, "Services chưa sẵn sàng. Check log server.")

    try:
        # Đọc audio 1 lần
        audio_bytes = await audio.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        y_normalized = preprocess_audio(y, sr=sr)
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, y_normalized, sr, format='WAV')
        wav_buffer.seek(0)
        clean_audio_bytes = wav_buffer.read()

        # 2. CHẠY GOP TRƯỚC (Để lấy Phoneme Score & Alignment)
        gop_result = gop_service.infer_gop(clean_audio_bytes, word)

        if "error" in gop_result:
            raise HTTPException(500, f"GOP Error: {gop_result['error']}")

        # 3. SMART CROP (Cắt audio dựa trên tai của GOP Model)
        # Mục đích: Cắt chính xác đoạn có giọng nói để thuật toán chia đều của Stress hoạt động tốt hơn

        # Lấy mốc thời gian từ GOP (nếu có)
        bounds = gop_result.get("speech_bounds")
        # Hoặc tính từ alignment nếu GOP trả về
        if not bounds and "alignment" in gop_result and gop_result["alignment"]:
            ali = gop_result["alignment"]
            # Lấy start của token đầu và end của token cuối, nới rộng ra 0.1s
            start_t = max(0, ali[0]["start"] - 0.1)
            end_t = min(len(y_normalized)/sr, ali[-1]["end"] + 0.1)
            bounds = {"start": start_t, "end": end_t}
        print(f"Bounds: {bounds}")

        if bounds:
            start_sample = int(bounds["start"] * sr)
            end_sample = int(bounds["end"] * sr)
            y_cropped = y_normalized[start_sample:end_sample]

        else:
            # Fallback nếu GOP không trả về bounds
            print("GOP did not return a bound")
            y_cropped = y_normalized

        # Encode bản đã cắt (cropped) để gửi vào Stress Service
        crop_buffer = io.BytesIO()
        sf.write(crop_buffer, y_cropped, sr, format='WAV')
        crop_buffer.seek(0)
        cropped_audio_bytes = crop_buffer.read()

        # 4. CHẠY STRESS DETECTION (Trên file audio đã được Smart Crop)
        stress_result = stress_service.predict(cropped_audio_bytes, word)

        if "error" in stress_result:
            raise HTTPException(500, f"Stress Error: {stress_result['error']}")

        # --- TỔNG HỢP KẾT QUẢ (Logic cũ giữ nguyên) ---

        # GOP Scores
        phones_score = {}
        for k, v in gop_result['details'].items():
            phones_score[k] = v['gop_score']

        # Overall Score
        avg_gop = gop_result['average_gop']
        overall_score = max(0, min(100, np.exp(avg_gop) * 100))

        # Stress Processing
        truth_stress = get_truth_stress(word)
        pred_probs = stress_result['raw_scores']
        pred_stress = [0] * len(pred_probs)

        if pred_probs:
            max_idx = np.argmax(pred_probs)
            pred_stress[max_idx] = 1

        # Fix length mismatch
        target_len = len(truth_stress)
        current_len = len(pred_stress)
        if current_len > target_len:
            pred_stress = pred_stress[:target_len]
        elif current_len < target_len:
            pred_stress = pred_stress + [0] * (target_len - current_len)

        return {
            "status": "success",
            "word": word,
            "phones": phones_score,
            "stress": {
                "truth": truth_stress,
                "infer": pred_stress,
                "confidence": stress_result['stress_probability'],
                "syllable_count": target_len
            },
            "overall_score": round(overall_score, 1)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
