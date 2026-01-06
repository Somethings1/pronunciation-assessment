import tensorflow as tf
import os

# Đường dẫn file .h5 gốc (cái file mày vừa train 1 tiếng ấy)
OLD_PATH = "checkpoints/stress_model.h5"
NEW_PATH = "checkpoints/stress_model.keras"

print(f"🚑 Đang giải cứu binh nhì Ryan: {OLD_PATH}")

if not os.path.exists(OLD_PATH):
    print(f"❌ Không tìm thấy file {OLD_PATH}. Mày check lại xem file model gốc nằm đâu?")
    exit()

try:
    # 1. Load model lên (Python script thường load được ngon ơ)
    model = tf.keras.models.load_model(OLD_PATH)
    print("✅ Đã load được model vào RAM.")

    # 2. Lưu lại sang định dạng .keras (Siêu bền)
    model.save(NEW_PATH)
    print(f"🎉 XONG! Đã lưu bản mới tại: {NEW_PATH}")
    print("👉 Giờ mày sửa app/main.py trỏ vào file .keras này là bao chạy!")

except Exception as e:
    print(f"❌ Lỗi cứu hộ: {e}")
