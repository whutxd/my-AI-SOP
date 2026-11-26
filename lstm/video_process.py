import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# ===================== 【你只需要改这里】 =====================
VIDEO_ROOT = r"D:\FDM\AI-SOP-main\lstm_runs_fine\data\video\cup"          # 视频根目录（下有class_id子文件夹）
OUTPUT_DIR = r"D:\FDM\AI-SOP-main\lstm_runs_fine\data\npy"         # 输出 x.npy y.npy
YOLO_WEIGHT = r"yolov5/yolov5m.pt"       # 你的YOLOv5权重路径
TARGET_FRAMES = 96                # 每个视频固定48帧
FEATURE_DIM = 131
# =============================================================

# ---------------------- 加载你的 YOLOv5 ----------------------
model = torch.hub.load('yolov5', 'custom', path=YOLO_WEIGHT, source='local')
model.conf = 0.25
model.classes = [41]  # 你指定的类别


# ---------------------- 加载 MediaPipe 双手 -------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_yolo5_features(frame):
    """
    从 YOLOv5 获取 5 维：[conf, cx_norm, cy_norm, w_norm, h_norm]
    只返回置信度最大的 1 个目标
    """
    h_img, w_img = frame.shape[:2]
    results = model(frame)  # YOLO 推理
    det = results.xyxyn[0]  # 归一化坐标 (x1,y1,x2,y2,conf,cls)

    if len(det) == 0:
        return np.zeros(5, dtype=np.float32)
    if len(det) > 0:
        det = det[:1]
    # 取最大置信度目标
    max_conf_idx = torch.argmax(det[:, 4]).item()
    x1, y1, x2, y2, conf, cls = det[max_conf_idx].tolist()

    # YOLO 格式转换为 cx cy w h (归一化)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1

    return np.array([conf, cx, cy, w, h], dtype=np.float32)

def get_hand_features(frame):
    """获取双手 126 维：左手63 + 右手63"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    left = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)

    if res.multi_hand_landmarks and res.multi_handedness:
        for i, lm in enumerate(res.multi_hand_landmarks):
            label = res.multi_handedness[i].classification[0].label
            kps = [p for pt in lm.landmark for p in (pt.x, pt.y, pt.z)]
            if label == "Left" and len(kps) == 63:
                left = np.array(kps, dtype=np.float32)
            if label == "Right" and len(kps) == 63:
                right = np.array(kps, dtype=np.float32)
    return left, right

def extract_131_features(frame):
    """【最终正确】131维 = YOLO5维 + 左手63 + 右手63"""
    yolo5 = get_yolo5_features(frame)
    left, right = get_hand_features(frame)
    feat = np.concatenate([yolo5, left, right])  # 5+63+63=131
    return feat

def sample_48_frames(video_path):
    """每个视频均匀采样 48 帧 → 返回 (48,131)"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    feats = []

    if total <= 0:
        cap.release()
        print(f"❌ 读取视频失败: {video_path}")
        return np.zeros((TARGET_FRAMES, FEATURE_DIM), dtype=np.float32)

    # 均匀取 48 帧索引
    indices = np.linspace(0, total-1, TARGET_FRAMES, dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"❌ 读取帧失败: {video_path}")
            feats.append(np.zeros(FEATURE_DIM, dtype=np.float32))
            continue
        feats.append(extract_131_features(frame))

    cap.release()
    return np.array(feats, dtype=np.float32)

def build_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X_list, y_list = [], []

    # 读取所有 class_id 文件夹
    class_dirs = sorted([d for d in os.listdir(VIDEO_ROOT)
                        if os.path.isdir(os.path.join(VIDEO_ROOT, d))])

    print(f"✅ 找到 {len(class_dirs)} 个类别: {class_dirs}")

    # 遍历每个类别
    for label, cls_name in enumerate(tqdm(class_dirs, desc="总类别")):
        cls_path = os.path.join(VIDEO_ROOT, cls_name)
        videos = [f for f in os.listdir(cls_path) if f.endswith(('mp4','avi','mov','mkv'))]

        for vid in tqdm(videos, desc=f"类别 {cls_name}"):
            vid_path = os.path.join(cls_path, vid)
            try:
                seq = sample_48_frames(vid_path)
                X_list.append(seq)
                y_list.append(label)
            except Exception as e:
                print(f"❌ 处理失败: {vid_path}, {str(e)}")

    # 保存最终数据集
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=int)
    np.save(os.path.join(OUTPUT_DIR, "x.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    print("\n🎉 数据集生成完成！")
    # print(f"x.npy shape: {X.shape}  (样本数, 48帧, 131维)")
    # print(f"y.npy shape: {y.shape}  (类别标签)")

if __name__ == "__main__":
    build_dataset()