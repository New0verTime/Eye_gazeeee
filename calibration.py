import pygame
import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import math
from utils import compute_head_pupil_metrics
poly1 = PolynomialFeatures(degree=2, include_bias=False)
#poly2 = PolynomialFeatures(degree=2, include_bias=False)

model_x = make_pipeline(poly1, LinearRegression())
from sklearn.ensemble import RandomForestRegressor
model_y = RandomForestRegressor(n_estimators=100, random_state=42)
# Thêm phần khởi tạo buffer dự đoán ở đầu chương trình
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

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY = (200, 200, 200)

# Thiết lập lưới
grid_size = 3
cell_width = actual_width / grid_size
cell_height = actual_height / grid_size

# Danh sách các điểm trong lưới
points = [(i, j) for i in range(grid_size) for j in range(grid_size)]

# Font chữ
font = pygame.font.SysFont('Arial', 30)

# Danh sách các hướng đầu
head_directions = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Center"]
current_direction_index = 0
current_point_index = 0
calibration_data1 = []
calibration_data2 = []
calibrated = False



def draw_grid():
    pygame.draw.line(screen, GRAY, (1, 0), (1, actual_height), 2)
    pygame.draw.line(screen, GRAY, (0, 1), (actual_width, 1), 2)
    for i in range(1, grid_size):
        x_pos = int(i * cell_width)
        y_pos = int(i * cell_height)
        pygame.draw.line(screen, GRAY, (x_pos, 0), (x_pos, actual_height), 2)
        pygame.draw.line(screen, GRAY, (0, y_pos), (actual_width, y_pos), 2)
    pygame.draw.line(screen, GRAY, (actual_width, 0), (actual_width, actual_height), 2)
    pygame.draw.line(screen, GRAY, (0, actual_height), (actual_width, actual_height), 2)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if current_direction_index < len(head_directions) and current_point_index < len(points):
                # Bước 1: Thu thập 50 mẫu vào 2 mảng tạm thời
                temp_data1 = []
                temp_data2 = []

                for i in range(50):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    h, w, _ = frame.shape
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(frame_rgb)
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            landmarks = face_landmarks.landmark
                            Hf, Wf, R_X, R_Y, _, _, h, v, EAR = compute_head_pupil_metrics(landmarks, frame.shape)
                            current_point = points[current_point_index]
                            center_x = current_point[0] * cell_width + cell_width / 2
                            center_y = current_point[1] * cell_height + cell_height / 2
                            
                            temp_data1.append([R_Y, h, center_x])
                            temp_data2.append([R_X, v, v*v, EAR, EAR*EAR, v*EAR, center_y])
                            print(f"[{i+1}/50] Point {current_point_index + 1}/9 | "
                                f"R_Y: {R_Y:.6f}, h: {h:.6f}, x: {center_x:.1f} | "
                                f"R_X: {R_X:.6f}, v: {v:.6f}, EAR: {EAR:.6f}, y: {center_y:.1f}")

                # Bước 2: Chia thành 5 nhóm và lấy trung bình từng nhóm
                for i in range(0, 50, 10):
                    group1 = np.array(temp_data1[i:i+10])
                    group2 = np.array(temp_data2[i:i+10])
                    
                    mean1 = np.mean(group1, axis=0)
                    mean2 = np.mean(group2, axis=0)
                    
                    calibration_data1.append(mean1.tolist())
                    calibration_data2.append(mean2.tolist())
                current_point_index += 1
                # Chuyển hướng đầu sau khi hoàn thành 9 điểm
                if current_point_index >= len(points):
                    current_point_index = 0
                    current_direction_index += 1
                    # Sau khi thu thập đủ 36 điểm, huấn luyện mô hình
                    if current_direction_index >= len(head_directions):
                        data_array1 = np.array(calibration_data1)
                        data_array2 = np.array(calibration_data2)
                        np.save('calibration_data1.npy', np.array(calibration_data1))
                        np.save('calibration_data2.npy', np.array(calibration_data2))
                        X = data_array1[:, :2]
                        y_x = data_array1[:, 2]
                        Y = data_array2[:, :6]
                        y_y =data_array2[:, 6]
                        model_x.fit(X, y_x)
                        model_y.fit(Y, y_y)
                        calibrated = True
                        print("Calibration completed! Starting prediction...")

    # Vẽ giao diện
    screen.fill(WHITE)
    draw_grid()
    
    if not calibrated and current_direction_index < len(head_directions):
        current_point = points[current_point_index]
        center_x = int(current_point[0] * cell_width + cell_width / 2)
        center_y = int(current_point[1] * cell_height + cell_height / 2)
        pygame.draw.circle(screen, RED, (center_x, center_y), 20)
        
        instruction = f"Direction: {head_directions[current_direction_index]}. At {current_point_index + 1}/9. Look at the red dot and click"
        text = font.render(instruction, True, BLACK)
        screen.blit(text, (20, 20))
    elif calibrated:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                Hf, Wf, R_X, R_Y, _, _, h, v, EAR = compute_head_pupil_metrics(landmarks, frame.shape)
                features1 = np.array([[R_Y, h]])
                features2 = np.array([[R_X, v, v*v, EAR, EAR*EAR, v*EAR]])

                # Dự đoán tọa độ mới
                new_x_pred = model_x.predict(features1)[0]
                new_y_pred = model_y.predict(features2)[0]
                
                # Thêm vào buffer
                x_pred_buffer.append(new_x_pred)
                y_pred_buffer.append(new_y_pred)
                
                # Giữ kích thước buffer
                if len(x_pred_buffer) > PREDICTION_BUFFER_SIZE:
                    x_pred_buffer.pop(0)
                    y_pred_buffer.pop(0)
                
                # Tính trung bình
                if len(x_pred_buffer) > 0:
                    avg_x = np.mean(x_pred_buffer)
                    avg_y = np.mean(y_pred_buffer)
                    
                    # Giới hạn trong màn hình
                    avg_x = np.clip(avg_x, 0, actual_width)
                    avg_y = np.clip(avg_y, 0, actual_height)
                    
                    # Vẽ điểm dự đoán trung bình
                    pygame.draw.circle(screen, RED, (int(avg_x), int(avg_y)), 20)
                    
                    # Hiển thị thông tin
                    pred_text = (f"Predicted (avg): ({avg_x:.1f}, {avg_y:.1f}) | "
                                f"R_X: {R_X/Hf:.3f}, R_Y: {R_Y/Hf:.3f}, h: {h/Hf:.3f}, v: {v/Hf:.3f}")
                    text = font.render(pred_text, True, BLACK)
                    screen.blit(text, (20, 20))
    else:
        completion_text = font.render("Calibration Completed. Press ESC to exit", True, BLACK)
        screen.blit(completion_text, (actual_width // 2 - 250, actual_height // 2))
    
    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

# Theo kiểm thử, feature R_X cho hướng y chạy khá kém, cần nghiên cứu thêm (M_I score 0.259)
# dù không quá liên quan, tuy nhiên head_eye_sum được 0.41
# sum_angles cho tọa độ bên x được MI 0.9908, tuy nhiên ko nên thêm vào (a(u+v) + bu + cv = (a+b)u + (a+c)v)
