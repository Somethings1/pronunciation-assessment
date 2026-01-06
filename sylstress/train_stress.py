import numpy as np
import json
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Masking, LSTM, Attention, Input, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# --- CẤU HÌNH ---
DATA_DIR = "./dataprep/data_train_ready"
MODEL_SAVE_PATH = "../server/models/sylstress/stress_model.keras"
SCALER_SAVE_PATH = "../server/models/sylstress/scaler_params.json"
ORIGINAL_DIM = 38 # Số lượng features (Acoustic)
MAX_SEQ_LEN = 10  # Độ dài chuỗi tối đa lúc training

def load_and_split_data():
    print("⏳ Đang load data lên RAM...")

    try:
        X = np.load(os.path.join(DATA_DIR, "X_train.npy")) # Shape (N, 10, 38)
        Y = np.load(os.path.join(DATA_DIR, "Y_train.npy")) # Shape (N, 10, 1)
    except FileNotFoundError:
        print(f"❌ Không thấy file data trong {DATA_DIR}. Chạy dataprep chưa đại ca?")
        exit()

    print(f"✅ Data gốc: {X.shape}")

    # --- 1. CHIA TẬP TRAIN / TEST (80/20) ---
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # --- 2. TÍNH MEAN/STD "SẠCH" (BỎ QUA PADDING) ---
    print("🧮 Đang tính toán thống kê (Bỏ qua padding = 0)...")

    # Tạo mask để biết đâu là data thật (những dòng mà không phải toàn số 0)
    # Giả sử nếu tất cả features đều là 0 thì đó là padding
    train_mask = np.any(X_train != 0, axis=2) # Shape (N_train, 10)

    # Lấy các vector features thật ra để tính toán
    # Flatten mảng mask để lấy đúng các phần tử tương ứng
    X_train_valid = X_train[train_mask] # Shape (Số lượng âm tiết thật, 38)

    mean_vec = np.mean(X_train_valid, axis=0)
    std_vec = np.std(X_train_valid, axis=0)

    # Lưu scaler để API dùng
    scaler_data = {
        "mean": mean_vec.tolist(),
        "std": std_vec.tolist()
    }
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    with open(SCALER_SAVE_PATH, 'w') as f:
        json.dump(scaler_data, f)
    print(f"💾 Đã lưu Scaler xịn vào {SCALER_SAVE_PATH}")

    # --- 3. CHUẨN HÓA DATA (GIỮ NGUYÊN PADDING LÀ 0) ---
    # Ta cần giữ giá trị 0.0 ở padding để lớp Masking của Keras hoạt động
    # Nên không được normalize kiểu broadcating mù quáng

    X_train_norm = np.copy(X_train)
    X_test_norm = np.copy(X_test)

    # Chỉ trừ mean/chia std ở những chỗ có dữ liệu thật
    # (X - Mean) / (Std + epsilon)
    X_train_norm[train_mask] = (X_train[train_mask] - mean_vec) / (std_vec + 1e-7)

    # Làm tương tự với tập test (dùng mean/std của tập train)
    test_mask = np.any(X_test != 0, axis=2)
    X_test_norm[test_mask] = (X_test[test_mask] - mean_vec) / (std_vec + 1e-7)

    print(f"🚀 Train set prepared: {X_train_norm.shape}")
    print(f"🧐 Test set prepared:  {X_test_norm.shape}")

    return X_train_norm, Y_train, X_test_norm, Y_test

def build_model(input_shape):
    # Input
    inputs = Input(shape=input_shape)

    # 1. MASKING LAYER: QUAN TRỌNG NHẤT
    # Nó bảo model: "Thấy số 0.0 thì lờ đi, đừng học, đừng tính loss"
    masked = Masking(mask_value=0.0)(inputs)

    # 2. LSTM Layers (Theo kiến trúc Paper Interspeech 2024 )
    # Paper dùng 3 lớp LSTM: 64 -> 32 -> 16
    lstm1 = LSTM(64, return_sequences=True)(masked)
    lstm1 = Dropout(0.3)(lstm1) # Dropout nhẹ để tránh overfit

    lstm2 = LSTM(32, return_sequences=True)(lstm1)
    lstm2 = Dropout(0.3)(lstm2)

    # Layer cuối cùng (Query Source) - 16 units
    lstm3 = LSTM(16, return_sequences=True)(lstm2)

    # 3. ATTENTION MECHANISM
    # Paper mô tả: "LSTM (16 cells) used as query, dense layer (4 units) used as value"
    # Để đơn giản và hiệu quả trong Keras: Ta chiếu LSTM output xuống không gian 4 chiều làm Value/Key
    # Và dùng chính LSTM output (hoặc chiếu xuống) làm Query.

    # Value projection (nén thông tin xuống 4 chiều như paper)
    value_proj = Dense(4, activation='tanh')(lstm3)
    # Query projection (để match dimension với Value khi tính dot product)
    query_proj = Dense(4, activation='tanh')(lstm3)

    # Attention Layer
    # Output sẽ là weighted sum của Value dựa trên độ tương đồng giữa Query và Key (ở đây Key=Value)
    attention = Attention()([query_proj, value_proj])

    # 4. OUTPUT LAYER
    # Dense 1 unit với Sigmoid activation cho bài toán Binary Classification
    outputs = Dense(1, activation='sigmoid', name='stress')(attention)

    model = Model(inputs, outputs)
    return model

def train():
    X_train, Y_train, X_test, Y_test = load_and_split_data()

    BATCH_SIZE = 64
    EPOCHS = 30 # Tăng lên chút vì model nhỏ hơn, hội tụ nhanh hơn

    # Build model với shape cố định lúc train cho tối ưu
    model = build_model((MAX_SEQ_LEN, ORIGINAL_DIM))

    model.compile(
        optimizer=Adam(learning_rate=0.001), # Paper dùng Adam [cite: 144]
        loss='binary_crossentropy',          # Paper dùng Binary Cross-Entropy [cite: 147]
        metrics=['accuracy']
    )

    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', verbose=1),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
    ]

    print("🔥 Bắt đầu training (Sequential Model - Attention)...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    # --- SAVING DYNAMIC MODEL ---
    print("💾 Đang convert sang Dynamic Model (cho input độ dài bất kỳ)...")
    dynamic_model = build_model((None, ORIGINAL_DIM))
    dynamic_model.set_weights(model.get_weights())
    dynamic_model.save(MODEL_SAVE_PATH)

    print(f"🎉 Xong! Model lưu tại: {MODEL_SAVE_PATH}")
    print("👉 Lưu ý: Khi dùng API, nhớ dùng scaler_params.json để normalize y hệt như lúc train!")

if __name__ == "__main__":
    train()
