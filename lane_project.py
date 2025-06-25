import numpy as np
import cv2
from pathlib import Path
import os


"""此份code為單張投影測試用"""

# === 1. 參數 ===
base_dir = Path(__file__).parent
scene_dir = base_dir / "highway_cloudy_day_2024-07-03-16-35-57"
txt_path = scene_dir / "Mobileye_q4" / "000000.txt"
image_path = scene_dir / "image" / "000000.png"
output_path = base_dir / "output" / "lane_overlay_000000.png"
os.makedirs(output_path.parent, exist_ok=True)



# === 2. 相機內參與外參 ===


K = np.array([
    [1418.667, 0.0, 640.0],
    [0.0, 1418.667, 360.0],
    [0.0, 0.0, 1.0]
])

T_q4_to_cam = np.array([
    [0.019606, 0.999807,  0.000834,   0.070000],
    [-0.084922, 0.000834, 0.996387,  1.340000],
    [ 0.996195, -0.019606, 0.084922, -1.150000],
    [0, 0, 0, 1]
])




# T_q4_to_cam = np.array([
#     [ 0.037, -0.999,  0.009,  0.0],
#     [-0.094, -0.012, -0.996, -0.3],
#     [ 0.995,  0.036, -0.15,  -0.43],  # ⬅ 調這裡
#     [ 0.0,    0.0,    0.0,    1.0]
# ])

# T_q4_to_cam = np.array([
#     [ 1.0,   0.017,  0.0,    0.1   ],
#     [-0.001, 0.07,  -0.998, -1.28 ],
#     [-0.017, 0.997,  0.07,  -0.865],
#     [ 0.0,   0.0,    0.0,    1.0   ]
# ])


# === 3. 讀影像 ===
img = cv2.imread(str(image_path))
assert img is not None, f" 圖片讀取失敗: {image_path}"
H, W = img.shape[:2]

# === 4. 解析 Mobileye 車道線 ===
with open(txt_path, "r") as f:
    lines = f.readlines()

lane_polys = []
i = 0
while i < len(lines) - 1:
    meta_line = lines[i].strip()
    coef_line = lines[i + 1].strip()

    try:
        meta = list(map(float, meta_line.split(",")))
        coef = list(map(float, coef_line.split(",")))
    except Exception as e:
        print(f" 第 {i} 行解析失敗：{e}")
        i += 2
        continue

    if len(meta) < 5:
        print(f"第 {i} 行 meta 長度不足，跳過：{meta_line}")
        i += 2
        continue

    confidence = meta[1]
    length = meta[4]
    if confidence < 0.5:
        i += 2
        continue

    lane_polys.append((coef, length))
    i += 2


# === 5. 繪製每條 lane ===


for coef, length in lane_polys:
    x_vals = np.linspace(0, length, 100)
    y_vals = sum(c * x_vals**i for i, c in enumerate(coef))
    z_vals = np.zeros_like(x_vals)
    pts_ego = np.stack([x_vals, y_vals, z_vals], axis=1)

    # 車體 → 相機
    RT = T_q4_to_cam
    pts_homo = np.hstack([pts_ego, np.ones((pts_ego.shape[0], 1))])
    pts_cam = (RT @ pts_homo.T).T[:, :3]

    # 過濾 Z <= 0
    mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask]

    if len(pts_cam) == 0:
        continue

    # 相機 → 影像
    proj = (K @ pts_cam.T).T
    proj_2d = proj[:, :2] / proj[:, 2:]

    # 畫線
    pts_img = proj_2d.astype(np.int32)
    cv2.polylines(img, [pts_img], isClosed=False, color=(0, 255, 0), thickness=4)

# === 6. 儲存輸出 ===
cv2.imwrite(str(output_path), img)
print(f" 已輸出：{output_path}")
