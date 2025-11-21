import uvicorn
import os
import numpy as np
import re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from g2p_en import G2p

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

        # --- 1. CHẠY STRESS DETECTION ---
        stress_result = stress_service.predict(audio_bytes, word)

        if "error" in stress_result:
            raise HTTPException(500, f"Stress Error: {stress_result['error']}")

        # Lấy Ground Truth Stress (Chuẩn từ điển)
        truth_stress = get_truth_stress(word)

        # Lấy Predicted Stress (Người dùng nói)
        # stress_result['raw_scores'] là xác suất từng âm tiết (vd: [0.1, 0.9, 0.05])
        # Ta convert sang binary [0, 1, 0] dựa trên argmax
        pred_probs = stress_result['raw_scores']
        pred_stress = [0] * len(pred_probs)

        # Tìm đỉnh cao nhất -> Gán là 1 (Primary Stress)
        # Lưu ý: Nếu từ điển có nhiều trọng âm (rare), logic này chỉ bắt Primary
        if pred_probs:
            max_idx = np.argmax(pred_probs)
            pred_stress[max_idx] = 1

        # Xử lý trường hợp lệch độ dài (do thuật toán tách âm tiết khác nhau)
        # Ưu tiên độ dài của Truth (từ điển)
        target_len = len(truth_stress)
        current_len = len(pred_stress)

        if current_len > target_len:
            pred_stress = pred_stress[:target_len]
        elif current_len < target_len:
            pred_stress = pred_stress + [0] * (target_len - current_len)

        # --- 2. CHẠY GOP (PHONEME SCORE) ---
        gop_result = gop_service.infer_gop(audio_bytes, word)

        if "error" in gop_result:
            raise HTTPException(500, f"GOP Error: {gop_result['error']}")

        # Clean up GOP structure
        # Chỉ lấy những info cần thiết cho FE
        phones_score = {}
        for k, v in gop_result['details'].items():
            # k dạng "AH_1", v['gop_score'] là số âm
            phones_score[k] = v['gop_score']

        # Tính điểm tổng (Overall)
        # Map GOP average (-5 đến 0) sang thang 0-100
        avg_gop = gop_result['average_gop']
        # Công thức heuristic: Score ~ exp(avg_gop) * 100
        overall_score = max(0, min(100, np.exp(avg_gop) * 100))

        # --- 3. TRẢ VỀ KẾT QUẢ ---
        return {
            "status": "success",
            "word": word,
            "phones": phones_score, # { "AH_0": -0.5, "P_1": -1.2 }
            "stress": {
                "truth": truth_stress, # [0, 1, 0]
                "infer": pred_stress,  # [0, 1, 0]
                "confidence": stress_result['stress_probability'], # 0.95
                "syllable_count": target_len
            },
            "overall_score": round(overall_score, 1) # 85.5
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
