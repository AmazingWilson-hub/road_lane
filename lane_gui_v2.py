import numpy as np
import cv2
import tkinter as tk
from tkinter import Scale, HORIZONTAL, Entry
from pathlib import Path
import os

"""此份code為 GUI 介面，調整車道線投影參數，最後可生出fine tune的外參矩陣"""

# === 固定參數 ===
base_dir = Path(__file__).parent
scene_dir = base_dir / "highway_cloudy_day_2024-07-03-16-35-57"
txt_path = scene_dir / "Mobileye_q4" / "000000.txt"
image_path = scene_dir / "image" / "000000.png"

K = np.array([
    [1418.667, 0.0, 640.0],
    [0.0, 1418.667, 360.0],
    [0.0, 0.0, 1.0]
])

def load_lane_polynomials():
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
        lane_polys.append((coef, meta[4], int(meta[2])))  # 加入 side 資訊
        i += 2
    return lane_polys


"""
| 參數名稱      | 單位             | 功能說明                                    |
| --------- | --------------    | ----------------------------              |
| `TX (dm)` | 分公尺(decimeter)  | 車體 → 相機的 **X 軸平移**（左右移動）     |
| `TY (dm)` | 分公尺(decimeter)  | 車體 → 相機的 **Y 軸平移**（上下移動）     |
| `TZ (dm)` | 分公尺(decimeter)  | 車體 → 相機的 **Z 軸平移**（前後移動）     |
| `Roll`    | 度(degree)        | 車體 → 相機的 **X 軸旋轉**（像是頭歪掉）    |
| `Pitch`   | 度(degree)        | 車體 → 相機的 **Y 軸旋轉**（抬頭或低頭）    |
| `Yaw`     | 度(degree)        | 車體 → 相機的 **Z 軸旋轉**（左右旋轉，像轉頭） |


"""

def create_transform(tx, ty, tz, roll, pitch, yaw):
    # 平移（公尺轉換）
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]

    # 歐拉角（deg → rad）
    rx, ry, rz = np.radians([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx

    T[:3, :3] = R
    return T

def update_from_entry(entry, slider):
    try:
        val = int(entry.get())
        slider.set(val)
        update_display()
    except ValueError:
        pass







def update_display():
    # 取得滑桿值
    tx = tx_slider.get() / 100.0
    ty = ty_slider.get() / 100.0
    tz = tz_slider.get() / 100.0
    roll = roll_slider.get()
    pitch = pitch_slider.get()
    yaw = yaw_slider.get()

    # 更新 entry 顯示
    tx_entry.delete(0, tk.END); tx_entry.insert(0, str(tx_slider.get()))
    ty_entry.delete(0, tk.END); ty_entry.insert(0, str(ty_slider.get()))
    tz_entry.delete(0, tk.END); tz_entry.insert(0, str(tz_slider.get()))
    roll_entry.delete(0, tk.END); roll_entry.insert(0, str(roll_slider.get()))
    pitch_entry.delete(0, tk.END); pitch_entry.insert(0, str(pitch_slider.get()))
    yaw_entry.delete(0, tk.END); yaw_entry.insert(0, str(yaw_slider.get()))

    # 複製原圖
    img = cv2.imread(str(image_path))
    if img is None:
        print(" 圖片載入失敗")
        return

    # === 建立並儲存轉換矩陣 ===
    T = create_transform(tx, ty, tz, roll, pitch, yaw)
    np.savetxt("current_extrinsic.txt", T, fmt="%.6f")
    print(" 儲存外參矩陣至 current_extrinsic.txt")

    # === 投影車道線 ===
    for coef, length, side in lane_polys:
        x = np.linspace(0, length, 100)
        y = sum(c * x**i for i, c in enumerate(coef))
        z = np.zeros_like(x)
        pts_ego = np.stack([x, y, z], axis=1)
        pts_homo = np.hstack([pts_ego, np.ones((len(x), 1))])

        pts_cam = (T @ pts_homo.T).T[:, :3]
        mask = pts_cam[:, 2] > 0
        pts_cam = pts_cam[mask]
        if len(pts_cam) == 0:
            continue

        proj = (K @ pts_cam.T).T
        proj_2d = proj[:, :2] / proj[:, 2:]
        pts_img = proj_2d.astype(np.int32)

        if side == 1:
            color = (0, 0, 255)
        elif side == 2:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)

        cv2.polylines(img, [pts_img], False, color, 3)

    cv2.imshow("Lane Projection", img)
    cv2.waitKey(1)


# === 主程式 ===
lane_polys = load_lane_polynomials()

root = tk.Tk()
root.title("RPY GUI")

# 建立滑桿與輸入欄位
sliders = []
entries = []
labels = ["TX (dm)", "TY (dm)", "TZ (dm)", "Roll (deg)", "Pitch (deg)", "Yaw (deg)"]

def create_slider_with_entry(label_text, row):
    label = tk.Label(root, text=label_text)
    label.grid(row=row, column=0)
    slider = Scale(root, from_=-200, to=200, orient=HORIZONTAL, length=300, command=lambda x: update_display())
    slider.set(0)
    slider.grid(row=row, column=1)
    entry = Entry(root, width=5)
    entry.insert(0, "0")
    entry.grid(row=row, column=2)
    entry.bind("<Return>", lambda event, e=entry, s=slider: update_from_entry(e, s))
    return slider, entry

tx_slider, tx_entry = create_slider_with_entry("TX (dm)", 0)
ty_slider, ty_entry = create_slider_with_entry("TY (dm)", 1)
tz_slider, tz_entry = create_slider_with_entry("TZ (dm)", 2)
roll_slider, roll_entry = create_slider_with_entry("Roll (deg)", 3)
pitch_slider, pitch_entry = create_slider_with_entry("Pitch (deg)", 4)
yaw_slider, yaw_entry = create_slider_with_entry("Yaw (deg)", 5)

update_display()
root.mainloop()
cv2.destroyAllWindows()
