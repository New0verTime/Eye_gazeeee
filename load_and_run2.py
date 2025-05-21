import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import time
import matplotlib.pyplot as plt
import pygame
import cv2
import mediapipe as mp
from utils import compute_head_pupil_metrics
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Khởi tạo mô hình
poly1 = PolynomialFeatures(degree=2, include_bias=False)
model_x = make_pipeline(poly1, LinearRegression())

from sklearn.ensemble import RandomForestRegressor
model_y = RandomForestRegressor(n_estimators=100, random_state=42)
model_y2 = LinearRegression()

# Buffer dự đoán
PREDICTION_BUFFER_SIZE = 5
x_pred_buffer = []
y_pred_buffer = []

# Khởi tạo MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Load dữ liệu từ file .npy
data_array1 = np.load('calibration_data1.npy')
data_array2 = np.load('calibration_data2.npy')

# Tách features và labels
X = data_array1[:, :2]
y_x = data_array1[:, 2]
Y = data_array2[:, :6]
y_y = data_array2[:, 6]
# Tính tổng 2 cột của X và reshape thành cột vector (n_samples, 1)
sum_X = np.sum(X, axis=1).reshape(-1, 1)

# Thêm cột tổng vào Y (đảm bảo cùng số samples)
Y_extended = np.hstack((Y, sum_X))

# Huấn luyện các model
model_x.fit(X, y_x)
model_y.fit(Y_extended, y_y)  # Sử dụng Y đã được mở rộng
model_y2.fit(Y_extended, y_y)
data_array1 = np.load('calibration_data11.npy')
data_array2 = np.load('calibration_data22.npy')

# Tách features và labels
X = data_array1[:, :2]
y_x = data_array1[:, 2]
Y = data_array2[:, :6]
y_y = data_array2[:, 6]
# === Đánh giá lỗi ===
# Tính tổng 2 cột của X và reshape thành cột vector (n_samples, 1)
sum_X = np.sum(X, axis=1).reshape(-1, 1)

# Thêm cột tổng vào Y (đảm bảo cùng số samples)
Y_extended = np.hstack((Y, sum_X))

# Huấn luyện các model
y_x_pred = model_x.predict(X)
for i in range(10):
    y_y_pred = (i/10)*model_y.predict(Y_extended) + (1 -(i/10))*model_y2.predict(Y_extended)

    mae_y = np.mean(np.abs(y_y - y_y_pred))
    mse_y = np.mean((y_y - y_y_pred)**2)

    # === Vẽ biểu đồ lỗi ===
    plt.figure(figsize=(12, 5))

    plt.title("Lỗi dự đoán tọa độ Y")
    plt.plot(y_y, label='Thực tế', marker='o')
    plt.plot(y_y_pred, label='Dự đoán', marker='x')
    plt.fill_between(range(len(y_y)), y_y, y_y_pred, color='blue', alpha=0.2)
    plt.xlabel("Sample")
    plt.ylabel("Pixel Y")
    plt.legend()
    plt.grid(True)
    print("\nTỔNG LỖI TUYỆT ĐỐI:")
    print(f"Tổng lỗi X (SAE): {np.sum(np.abs(y_x - y_x_pred)):.2f}")
    print(f"Tổng lỗi Y (SAE): {np.sum(np.abs(y_y - y_y_pred)):.2f}")
    print(f"Tổng lỗi cả X và Y: {np.sum(np.abs(y_x - y_x_pred)) + np.sum(np.abs(y_y - y_y_pred)):.2f}")
    plt.tight_layout()
    plt.savefig(f"plot_y_blend_{i}.png")
    plt.close()             # Đóng plot cũ để tránh mở quá nhiều cửa sổ


# === Bắt đầu vòng lặp chính Pygame sau khi kiểm tra xong ===
