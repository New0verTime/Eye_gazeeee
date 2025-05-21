# thêm tính năng nếu nhắm mắt thì ko detect nữa
# hàm ear có thể tính bằng nhiều điểm hơn, đỡ nhiễu
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline
import pygame
import cv2
import mediapipe as mp
from utils import compute_head_pupil_metrics

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Khởi tạo mô hình
poly1 = PolynomialFeatures(degree=2, include_bias=False)
model_x = make_pipeline(poly1, LinearRegression())
model_y = RandomForestRegressor(n_estimators=100, random_state=42)

# Buffer dự đoán
PREDICTION_BUFFER_SIZE = 5
x_pred_buffer = []
y_pred_buffer = []

# Khởi tạo MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Khởi tạo Pygame
pygame.init()
info = pygame.display.Info()
actual_width, actual_height = info.current_w, info.current_h
screen = pygame.display.set_mode((actual_width, actual_height), pygame.FULLSCREEN)
actual_width, actual_height = screen.get_size()
pygame.display.set_caption("Eye Tracking Calibration")
font = pygame.font.SysFont('Arial', 30)

# Load dữ liệu từ file .npy
data_array1 = np.load('calibration_data1.npy')
data_array2 = np.load('calibration_data2.npy')

# Tách features và labels
X = data_array1[:, :2]
y_x = data_array1[:, 2]
Y = data_array2[:, :6]
y_y = data_array2[:, 6]

# Huấn luyện lại mô hình
model_x.fit(X, y_x)
model_y.fit(Y, y_y)
print("Models loaded and re-trained from saved data.")

# Main loop
while True:
    # Xử lý sự kiện Pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            cap.release()
            cv2.destroyAllWindows()
            exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                cap.release()
                cv2.destroyAllWindows()
                exit()

    # Đọc frame từ camera
    ret, frame = cap.read()
    if not ret:
        break

    screen.fill(WHITE)  # Làm mới màn hình mỗi vòng lặp

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            Hf, Wf, R_X, R_Y, _, _, h, v, EAR = compute_head_pupil_metrics(landmarks, frame.shape)

            features1 = np.array([[R_Y, h]])
            features2 = np.array([[R_X, v, v*v, EAR, EAR*EAR, v*EAR]])
            new_x_pred = model_x.predict(features1)[0]
            new_y_pred = model_y.predict(features2)[0]

            # Thêm vào buffer
            x_pred_buffer.append(new_x_pred)
            y_pred_buffer.append(new_y_pred)

            # Giữ kích thước buffer
            if len(x_pred_buffer) > PREDICTION_BUFFER_SIZE:
                x_pred_buffer.pop(0)
                y_pred_buffer.pop(0)

            # Tính trung bình và vẽ
            if len(x_pred_buffer) > 0:
                avg_x = np.mean(x_pred_buffer)
                avg_y = np.mean(y_pred_buffer)

                # Giới hạn trong màn hình
                avg_x = np.clip(avg_x, 0, actual_width)
                avg_y = np.clip(avg_y, 0, actual_height)

                # Vẽ điểm dự đoán trung bình
                pygame.draw.circle(screen, RED, (int(avg_x), int(avg_y)), 20)

                # Vẽ text nếu không chia 0
                if Hf != 0:
                    pred_text = (f"Predicted (avg): ({avg_x:.1f}, {avg_y:.1f}) | "
                                 f"R_X: {R_X/Hf:.3f}, R_Y: {R_Y/Hf:.3f}, h: {h/Hf:.3f}, v: {v/Hf:.3f}")
                    text = font.render(pred_text, True, BLACK)
                    screen.blit(text, (20, 20))

    pygame.display.flip()

# Thoát chương trình
pygame.quit()
cap.release()
cv2.destroyAllWindows()
