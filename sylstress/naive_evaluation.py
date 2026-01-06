import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import classification_report, confusion_matrix

# --- CẤU HÌNH ---
DATA_DIR = "./dataprep/data_train_ready"
MODEL_PATH = "../server/models/sylstress/stress_model.keras"
SCALER_PATH = "../server/models/sylstress/scaler_params.json"
TEST_SIZE = 0.2

# --- CẤU HÌNH GIẢ LẬP UNIFORM ---
# Giả sử features của ông: [Duration, Energy, Pitch, ...]
# Ông cần biết index của Duration trong 38 features để làm phẳng nó.
# Nếu ông không nhớ, tôi sẽ giả định index 0 là Duration (thường là thế).
# Nếu sai, ông sửa lại cái INDEX này nhé.
DURATION_FEATURE_IDX = 0
SMOOTHING_SIGMA = 1.5 # Độ nhoè biên giới (càng lớn càng giống chia đều mù quáng)

def load_scaler():
    if not os.path.exists(SCALER_PATH):
        print("❌ Mất file scaler rồi.")
        return None, None
    with open(SCALER_PATH, 'r') as f:
        data = json.load(f)
    return np.array(data['mean']), np.array(data['std'])

def get_real_length(x_sample):
    feature_sum = np.sum(np.abs(x_sample), axis=1)
    non_zero = np.where(feature_sum > 1e-5)[0]
    return non_zero[-1] + 1 if len(non_zero) > 0 else 0

def simulate_uniform_segmentation(X_original):
    """
    Biến đổi features xịn thành features 'chia đều' (bố đời).
    """
    print(f"🔨 Đang đập nát features theo phong cách 'Chia đều' (Sigma={SMOOTHING_SIGMA})...")
    X_bad = np.copy(X_original)

    for i in range(len(X_bad)):
        real_len = get_real_length(X_original[i])
        if real_len <= 1: continue

        # 1. XỬ LÝ DURATION: Chia đều
        # Trong thực tế, uniform segmentation khiến mọi âm tiết có duration như nhau
        # Ta lấy trung bình duration của cả từ và gán cho tất cả
        # (Lưu ý: X ở đây chưa normalize hoặc đã normalize đều được, vì ta làm phẳng)

        # Lấy feature cột Duration
        durations = X_bad[i, :real_len, DURATION_FEATURE_IDX]
        avg_duration = np.mean(durations)

        # Gán tất cả bằng trung bình -> Mất hoàn toàn thông tin tương phản Ngắn/Dài
        X_bad[i, :real_len, DURATION_FEATURE_IDX] = avg_duration

        # 2. XỬ LÝ ENERGY/PITCH: Làm mờ (Smearing)
        # Khi cắt sai, năng lượng của âm tiết Mạnh sẽ tràn sang âm tiết Yếu
        # Ta dùng Gaussian Filter để làm mờ dọc theo trục thời gian

        # Apply cho tất cả features trừ padding
        # sigma=1.5 mô phỏng việc cắt lẹm vào nhau khá nhiều
        for feat_idx in range(X_bad.shape[2]):
            if feat_idx == DURATION_FEATURE_IDX: continue # Duration đã xử lý riêng

            seq = X_bad[i, :real_len, feat_idx]
            # Làm mờ sequence
            blurred_seq = gaussian_filter1d(seq, sigma=SMOOTHING_SIGMA, mode='nearest')
            X_bad[i, :real_len, feat_idx] = blurred_seq

    return X_bad

def run_test():
    # 1. Load Data
    print("📦 Loading data...")
    X = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y = np.load(os.path.join(DATA_DIR, "Y_train.npy"))
    _, X_test, _, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=42)

    # 2. TẠO DATA "BỐ ĐỜI" (UNIFORM)
    # Lưu ý: Phải làm biến đổi TRƯỚC khi Normalize
    X_test_uniform = simulate_uniform_segmentation(X_test)

    # 3. Normalize (Dùng mean/std của tập train gốc)
    print("🧪 Normalizing...")
    mean_vec, std_vec = load_scaler()

    # Hàm normalize helper
    def normalize(data):
        d_norm = np.copy(data)
        mask = np.any(data != 0, axis=2)
        d_norm[mask] = (data[mask] - mean_vec) / (std_vec + 1e-7)
        return d_norm

    X_test_clean_norm = normalize(X_test)       # Data xịn (để đối chiếu)
    X_test_bad_norm   = normalize(X_test_uniform) # Data đểu

    # 4. Load Model
    print("🧠 Loading model...")
    model = load_model(MODEL_PATH)

    # 5. Predict & Compare
    print("\n⚔️  BATTLE: ALIGNED (XỊN) vs. UNIFORM (ĐỂU) ⚔️")

    # --- Đánh giá hàm helper ---
    def evaluate(X_input, name):
        probs = model.predict(X_input, verbose=0)
        correct = 0
        total = 0
        for i in range(len(X_input)):
            real_len = get_real_length(X_test[i]) # Lấy length từ data gốc
            if real_len == 0 or np.max(Y_test[i]) == 0: continue

            true_idx = np.argmax(Y_test[i][:real_len])
            pred_idx = np.argmax(probs[i][:real_len])

            if true_idx == pred_idx: correct += 1
            total += 1
        return (correct / total) * 100

    acc_clean = evaluate(X_test_clean_norm, "Aligned")
    acc_bad   = evaluate(X_test_bad_norm, "Uniform")

    print("="*40)
    print(f"✅ Accuracy với Time-Aligned (Xịn):  {acc_clean:.2f}%")
    print(f"❌ Accuracy với Uniform Split (Lởm): {acc_bad:.2f}%")
    print("-" * 40)
    print(f"📉 Mức độ tụt giảm: {acc_clean - acc_bad:.2f}%")
    print("="*40)

    if (acc_clean - acc_bad) > 15:
        print("💡 KẾT LUẬN: Đấy thấy chưa? Tôi đã bảo rồi!")
        print("   Mất thông tin Duration và làm nhòe Energy khiến model thành phế vật.")
    else:
        print("💡 KẾT LUẬN: Hơi bất ngờ, có thể feature Duration không quan trọng lắm?")
        print("   Hoặc Gaussian Filter chưa đủ mạnh để mô phỏng độ 'ngu' của Uniform split.")

if __name__ == "__main__":
    run_test()
