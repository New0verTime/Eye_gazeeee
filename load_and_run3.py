import numpy as np
import matplotlib.pyplot as plt
data_array1 = np.load('calibration_data1.npy')
data_array2 = np.load('calibration_data2.npy')
# Tách features và labels
X = data_array1[:, :2]
y_x = data_array1[:, 2]
Y = data_array2[:, :6]
y_y = data_array2[:, 6]
sum_X = np.sum(X, axis=1).reshape(-1, 1)

import matplotlib.pyplot as plt
import numpy as np

# Ghép tất cả feature + label lại thành một ma trận lớn
# Gồm: eye_diff_x, eye_diff_y, head_x, head_y, head_z, pupil_x, pupil_y, blink_ratio, sum_eye_diff, y_x, y_y
all_features = np.hstack((X, Y, sum_X, y_x.reshape(-1, 1), y_y.reshape(-1, 1)))

# Tên cột tương ứng
all_feature_names = [
    "head_x", "eye_diff_x",
    "head_y", "eye_diff_y", "eye_diff_y^2",
    "EAR", "EAR^2", "EAR*eye_diff",
    "sum_eye_head_x", "y_x", "y_y"
]

# Tính ma trận tương quan (Pearson)
corr_matrix = np.corrcoef(all_features, rowvar=False)

# Vẽ heatmap bằng matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
plt.colorbar(label="Correlation coefficient")

# Đánh dấu các cột hàng
plt.xticks(ticks=np.arange(len(all_feature_names)), labels=all_feature_names, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(all_feature_names)), labels=all_feature_names)

# Thêm nhãn hệ số vào ô (tùy chọn)
for i in range(len(all_feature_names)):
    for j in range(len(all_feature_names)):
        value = corr_matrix[i, j]
        plt.text(j, i, f"{value:.2f}", ha='center', va='center',
                 color='white' if abs(value) > 0.5 else 'black', fontsize=8)

plt.title("Ma trận hệ số tương quan giữa các đặc trưng và output")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()
