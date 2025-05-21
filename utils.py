import numpy as np
import math
def compute_head_pupil_metrics(landmarks, frame_shape):
    h, w, _ = frame_shape
    def get_xyz(idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y, lm.z if lm.z is not None else 0])
    
    PME = get_xyz(168)
    PBN = get_xyz(2)
    PlMCA = get_xyz(362)
    PrMCA = get_xyz(133)
    
    Hf = np.linalg.norm(PME - PBN)
    Wf = np.linalg.norm(PlMCA - PrMCA)
    R_X = (PME[2] - PBN[2]) / (PME[1] - PBN[1]) if PME[1] != PBN[1] else 0
    R_Y = (PlMCA[2] - PrMCA[2]) / (PlMCA[0] - PrMCA[0]) if PlMCA[0] != PrMCA[0] else 0
    L_height = np.linalg.norm(get_xyz(159) - get_xyz(145))
    L_width = np.linalg.norm(get_xyz(33) - get_xyz(133))
            
    R_height = np.linalg.norm(get_xyz(386) -get_xyz(374))
    R_width = np.linalg.norm(get_xyz(362) - get_xyz(263))
    EAR = (L_height*L_width + R_height*R_width)/(L_width+R_width)
    left_top = 223
    left_bottom = 230
    right_top = 443
    right_bottom = 450
    left_pupil = list(range(468, 473))
    right_pupil = list(range(473, 478))
    
    lp_y = np.mean([landmarks[i].y for i in left_pupil])
    rp_y = np.mean([landmarks[i].y for i in right_pupil])
    lt_y = landmarks[left_top].y
    lb_y = landmarks[left_bottom].y
    rt_y = landmarks[right_top].y
    rb_y = landmarks[right_bottom].y
    
    l33 = landmarks[33].x
    l133 = landmarks[133].x
    l468 = np.mean([landmarks[i].x for i in left_pupil])
    r362 = landmarks[362].x
    r263 = landmarks[263].x
    r473 = np.mean([landmarks[i].x for i in right_pupil])
    
    vertical_left = (lp_y - lt_y) / (lb_y - lt_y + 1e-6)
    vertical_right = (rp_y - rt_y) / (rb_y - rt_y + 1e-6)
    vertical_gaze = (vertical_left + vertical_right) / 2
    
    horizontal_left = (l468 - l33) / (l133 - l33 + 1e-6)
    horizontal_right = (r473 - r362) / (r263 - r362 + 1e-6)
    horizontal_gaze = (horizontal_left + horizontal_right) / 2
    
    sum_left_x = sum(landmarks[i].x for i in range(469, 473)) / 4
    sum_left_y = sum(landmarks[i].y for i in range(469, 473)) / 4
    PlMCA = landmarks[362]  # Trung tâm mắt trái trong MediaPipe
    PrMCA = landmarks[133]  # Trung tâm mắt phải trong MediaPipe
    P_lP_x = (PlMCA.x - sum_left_x) / Wf
    P_lP_y = (PlMCA.y - sum_left_y) / Hf
    
    # Tính P_rP(x,y)
    sum_right_x = sum(landmarks[i].x for i in range(474, 478)) / 4
    sum_right_y = sum(landmarks[i].y for i in range(474, 478)) / 4
    P_rP_x = (PrMCA.x - sum_right_x) / Wf
    P_rP_y = (PrMCA.y - sum_right_y) / Hf
    
    # Tính P_P(x,y)
    P_P_x = P_lP_x + P_rP_x
    P_P_y = P_lP_y + P_rP_y
    
    return Hf, Wf, math.atan(R_X), math.atan(R_Y), horizontal_gaze, vertical_gaze, P_P_x, P_P_y, EAR