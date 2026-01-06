import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# --- CẤU HÌNH ---
DATA_DIR = "./dataprep/data_train_ready"
MODEL_PATH = "../server/models/sylstress/stress_model.keras"
SCALER_PATH = "../server/models/sylstress/scaler_params.json" # PHẢI CÓ CÁI NÀY
REPORT_DIR = "evaluation_reports"
TEST_SIZE = 0.2

def get_real_length(x_sample):
    """
    Tính độ dài thật. Vì lát nữa ta normalize nhưng vẫn giữ padding = 0,
    nên logic check sum > 0 vẫn hoạt động tốt.
    """
    feature_sum = np.sum(np.abs(x_sample), axis=1)
    non_zero_indices = np.where(feature_sum > 1e-5)[0]
    if len(non_zero_indices) == 0:
        return 0
    return non_zero_indices[-1] + 1

def load_scaler():
    """Load thông số Mean/Std đã lưu lúc train"""
    if not os.path.exists(SCALER_PATH):
        print(f"❌ Chết dở, không thấy file scaler tại {SCALER_PATH}!")
        print("Model sẽ bị ngu đi vì không được normalize.")
        return None, None

    with open(SCALER_PATH, 'r') as f:
        data = json.load(f)
    return np.array(data['mean']), np.array(data['std'])

def scientific_evaluation():
    if not os.path.exists(DATA_DIR):
        print(f"❌ Không tìm thấy data tại {DATA_DIR}")
        return

    # 1. Load Data Raw
    print("📦 Loading raw dataset...")
    X = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y = np.load(os.path.join(DATA_DIR, "Y_train.npy"))

    # Split y hệt lúc train để đảm bảo X_test khớp nhau
    # (Lưu ý: random_state phải giống hệt file train_stress.py)
    _, X_test, _, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=42)

    print(f"📊 Test set size: {len(X_test)} samples")

    # 2. NORMALIZE DATA (QUAN TRỌNG VÃI CHƯỞNG)
    print("🧪 Applying Normalization...")
    mean_vec, std_vec = load_scaler()

    if mean_vec is not None:
        # Tạo bản copy để không ghi đè data gốc (cần data gốc để check length)
        X_test_norm = np.copy(X_test)

        # Tìm những chỗ không phải padding
        mask = np.any(X_test != 0, axis=2)

        # Normalize: (X - Mean) / Std
        # Chỉ áp dụng lên những pixel thật, giữ nguyên số 0 ở padding
        X_test_norm[mask] = (X_test[mask] - mean_vec) / (std_vec + 1e-7)
    else:
        X_test_norm = X_test # Chấp nhận đau thương

    # 3. Load Model & Predict
    print("🧠 Loading model...")
    model = load_model(MODEL_PATH)

    print("🔮 Predicting...")
    Y_pred_prob = model.predict(X_test_norm, verbose=1)

    # 4. Xử lý số liệu
    y_true_flat = []    # List nhãn thật (0/1) để tính F1
    y_pred_flat = []    # List nhãn dự đoán (0/1) theo threshold

    word_correct_count = 0  # Số từ đúng trọng âm (theo Argmax)
    total_valid_words = 0

    print("🕵️ Analyzing results...")

    for i in range(len(X_test)):
        # Dùng X_test gốc (chưa norm) hoặc check trên Y để lấy length chuẩn
        # (Vì sau khi norm, giá trị có thể rất nhỏ nhưng khác 0, check trên gốc an toàn hơn)
        real_len = get_real_length(X_test[i])

        if real_len == 0: continue

        # --- LOGIC 1: Token Level (Cho Classification Report) ---
        # Cắt chuỗi
        true_seq = Y_test[i][:real_len].flatten()
        pred_prob_seq = Y_pred_prob[i][:real_len].flatten()

        # Threshold 0.5 cho từng âm tiết
        pred_bin_seq = (pred_prob_seq > 0.5).astype(int)

        y_true_flat.extend(true_seq)
        y_pred_flat.extend(pred_bin_seq)

        # --- LOGIC 2: Word Level (Cho Stress Accuracy chuẩn Paper) ---
        # Logic: Trong 1 từ, âm tiết nào có xác suất cao nhất -> Stress
        # So sánh vị trí Max của Pred vs vị trí Max của True

        # Nếu trong nhãn thật không có trọng âm (từ đơn âm tiết không stress hoặc lỗi data), bỏ qua
        if np.max(true_seq) == 0:
            continue

        true_stress_idx = np.argmax(true_seq)
        pred_stress_idx = np.argmax(pred_prob_seq)

        if true_stress_idx == pred_stress_idx:
            word_correct_count += 1

        total_valid_words += 1

    # 5. Báo cáo
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "scientific_report.txt")

    # Word Accuracy
    word_acc = (word_correct_count / total_valid_words) * 100 if total_valid_words > 0 else 0

    # Token Metrics
    cls_report = classification_report(y_true_flat, y_pred_flat, target_names=["Unstressed (0)", "Stressed (1)"], digits=4)
    cm = confusion_matrix(y_true_flat, y_pred_flat)

    output_str = "="*60 + "\n"
    output_str += "        BÁO CÁO ĐÁNH GIÁ MÔ HÌNH (FINAL)        \n"
    output_str += "="*60 + "\n"
    output_str += f"Total Valid Words: {total_valid_words}\n"
    output_str += f"🔥 Word-Level Accuracy (Argmax): {word_acc:.2f}%\n"
    output_str += "   (Tỷ lệ model chọn đúng vị trí trọng âm trong từ)\n"
    output_str += "-"*60 + "\n"
    output_str += "TOKEN-LEVEL METRICS (Threshold 0.5):\n"
    output_str += cls_report + "\n"
    output_str += "-"*60 + "\n"
    output_str += "Confusion Matrix:\n"
    output_str += str(cm) + "\n"

    print(output_str)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(output_str)

    # 6. Vẽ Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Unstressed", "Stressed"],
                yticklabels=["Unstressed", "Stressed"])
    plt.title(f'Confusion Matrix\nWord Acc: {word_acc:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    save_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"🖼 Đã lưu biểu đồ tại: {save_path}")

if __name__ == "__main__":
    scientific_evaluation()
