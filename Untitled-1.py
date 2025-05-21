import cv2
import mediapipe as mp
import numpy as np

def compute_head_pupil_metrics(landmarks, frame_shape):
    h, w, _ = frame_shape
    
    # Chuyển tọa độ 2D sang 3D dựa trên kích thước khung hình
    def get_xyz(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h, lm.z * w if lm.z is not None else 0])
    
    # --- Các điểm chính ---
    PME = get_xyz(168)  # Điểm giữa mắt
    PBN = get_xyz(2)    # Điểm mũi
    PlMCA = get_xyz(362)  # Điểm trung tâm mắt trái
    PrMCA = get_xyz(133)  # Điểm trung tâm mắt phải
    
    # --- Hf và Wf ---
    Hf = np.linalg.norm(PME - PBN)
    Wf = np.linalg.norm(PlMCA - PrMCA)
    
    # --- R_X và R_Y ---
    R_X = (PME[2] - PBN[2]) / (PME[1] - PBN[1]) if PME[1] != PBN[1] else 0
    R_Y = (PlMCA[2] - PrMCA[2]) / (PlMCA[0] - PrMCA[0]) if PlMCA[0] != PrMCA[0] else 0
    
    # --- Landmark indexes ---
    left_top = 223
    left_bottom = 230
    right_top = 443
    right_bottom = 450
    left_pupil = list(range(468, 473))
    right_pupil = list(range(473, 478))
    
    # --- Pupil vertical position ---
    lp_y = np.mean([landmarks[i].y for i in left_pupil])
    rp_y = np.mean([landmarks[i].y for i in right_pupil])
    lt_y = landmarks[left_top].y
    lb_y = landmarks[left_bottom].y
    rt_y = landmarks[right_top].y
    rb_y = landmarks[right_bottom].y
    
    # --- Horizontal position ---
    l33 = landmarks[33].x
    l133 = landmarks[133].x
    l468 = np.mean([landmarks[i].x for i in left_pupil])
    r362 = landmarks[362].x
    r263 = landmarks[263].x
    r473 = np.mean([landmarks[i].x for i in right_pupil])
    
    # --- Compute gaze ratios ---
    vertical_left = (lp_y - lt_y) / (lb_y - lt_y + 1e-6)
    vertical_right = (rp_y - rt_y) / (rb_y - rt_y + 1e-6)
    vertical_gaze = (vertical_left + vertical_right) / 2
    
    horizontal_left = (l468 - l33) / (l133 - l33 + 1e-6)
    horizontal_right = (r473 - r362) / (r263 - r362 + 1e-6)
    horizontal_gaze = (horizontal_left + horizontal_right) / 2
    
    return Hf, Wf, R_X, R_Y, horizontal_gaze, vertical_gaze

# Khởi tạo MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Mở camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            Hf, Wf, R_X, R_Y, h, v = compute_head_pupil_metrics(landmarks, frame.shape)
            print(f"Hf: {Hf:.2f}, Wf: {Wf:.2f}, R_X: {R_X:.2f}, R_Y: {R_Y:.2f}, h: {h:.2f}, v: {v:.2f}")
    
   