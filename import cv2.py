import cv2
import mediapipe as mp

# Khởi tạo Face Mesh với 478 điểm (cần model mới của MediaPipe)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Danh sách các điểm bạn muốn vẽ (ví dụ: 2 - nose tip, 33 - eye, 133 - eye, 168 - pupil, 362 - eye)
#landmark_ids_to_draw = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
landmark_ids_to_draw = [ 168 ] # bạn có thể sửa 385, 386, 387 | 380, 374, 373

# Mở webcam (hoặc dùng video bằng cách thay 0 bằng đường dẫn file)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx in landmark_ids_to_draw:
                if idx < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    print(f"idx:{idx}, z:{lm.z}")
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                    cv2.putText(frame, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imshow('Selected Landmarks', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
