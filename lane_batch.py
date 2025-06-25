import numpy as np
import cv2
from pathlib import Path
import os

"""此份code為批次處理多張圖片，並畫出車道線的投影"""

# === 相機內參與外參 ===
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

# === 路徑設定 ===
base_dir = Path(__file__).parent
scene_dir = base_dir / "highway_cloudy_day_2024-07-03-16-35-57"
img_dir = scene_dir / "image"
txt_dir = scene_dir / "Mobileye_q4"
out_dir = base_dir / "output_batch"
out_dir.mkdir(parents=True, exist_ok=True)

# === 批次處理圖片並畫車道線 ===
img_paths = sorted(img_dir.glob("*.png"))

for img_path in img_paths:
    frame_id = img_path.stem
    txt_path = txt_dir / f"{frame_id}.txt"
    if not txt_path.exists():
        print(f"❌ 缺少對應 txt：{txt_path}")
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ 圖片讀取失敗：{img_path}")
        continue

    # === 讀取對應 txt 車道資料 ===
    with open(txt_path, "r") as f:
        lines = f.readlines()

    lane_polys = []
    i = 0
    while i < len(lines) - 1:
        try:
            meta = list(map(float, lines[i].strip().split(",")))
            coef = list(map(float, lines[i + 1].strip().split(",")))
        except:
            i += 2
            continue
        if len(meta) < 5 or meta[1] < 0.5:
            i += 2
            continue
        lane_polys.append((coef, meta[4]))
        i += 2

    # === 投影每條車道線 ===
    for coef, length in lane_polys:
        x = np.linspace(0, length, 100)
        y = sum(c * x**i for i, c in enumerate(coef))
        z = np.zeros_like(x)
        pts_ego = np.stack([x, y, z], axis=1)
        pts_homo = np.hstack([pts_ego, np.ones((len(x), 1))])
        pts_cam = (T_q4_to_cam @ pts_homo.T).T[:, :3]

        mask = pts_cam[:, 2] > 0
        pts_cam = pts_cam[mask]
        if len(pts_cam) == 0:
            continue

        proj = (K @ pts_cam.T).T
        proj_2d = proj[:, :2] / proj[:, 2:]
        pts_img = proj_2d.astype(np.int32)

        cv2.polylines(img, [pts_img], False, (0, 255, 0), 3)

    out_path = out_dir / f"{frame_id}.png"
    cv2.imwrite(str(out_path), img)
    print(f"✅ 輸出：{out_path}")

# === 合成影片 ===
output_video = base_dir / "lane_output_video.mp4"
image_files = sorted(out_dir.glob("*.png"))
if len(image_files) == 0:
    print("⚠️ 沒有圖片輸出，無法產生影片")
    exit()

sample_img = cv2.imread(str(image_files[0]))
H, W = sample_img.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 10
video_writer = cv2.VideoWriter(str(output_video), fourcc, fps, (W, H))

for img_path in image_files:
    img = cv2.imread(str(img_path))
    video_writer.write(img)

video_writer.release()
print(f"🎥 已完成影片：{output_video}")
