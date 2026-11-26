import cv2
import random
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# ===================== 基础配置 =====================
# 旋转角度
ROTATE_ANGLE_RANGE = (-20, 20)
# 速度倍率：慢放 / 正常 / 快放
SPEED_RATIOS = [0.7, 1.0, 1.3]
# 帧交换概率
FRAME_SWAP_PROB = 0.2
# 丢帧概率
FRAME_DROP_PROB = 0.1
# 拼接帧数
CONCAT_FRAME_COUNT = 8

# ===================== 工具函数 =====================
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames, fps, w, h

def save_video(frames, fps, w, h, save_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()

# ===================== 一、空间增强（独立） =====================
def aug_flip(frames, fps, w, h):
    return [cv2.flip(f, 1) for f in frames], fps

def aug_crop_scale(frames, fps, w, h):
    new_frames = []
    scale = random.uniform(0.7, 1.0)
    new_h, new_w = int(h * scale), int(w * scale)
    x = random.randint(0, w - new_w)
    y = random.randint(0, h - new_h)
    for f in frames:
        crop = f[y:y+new_h, x:x+new_w]
        new_frames.append(cv2.resize(crop, (w, h)))
    return new_frames, fps

def aug_color_jitter(frames, fps, w, h):
    new_frames = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 2] *= random.uniform(0.85, 1.15)
        hsv[..., 1] *= random.uniform(0.85, 1.15)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        f = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        b, g, r = cv2.split(f)
        r = np.clip(r + random.randint(-8, 8), 0, 255)
        g = np.clip(g + random.randint(-8, 8), 0, 255)
        b = np.clip(b + random.randint(-8, 8), 0, 255)
        new_frames.append(cv2.merge((b, g, r)))
    return new_frames, fps

def aug_blur_noise(frames, fps, w, h):
    new_frames = []
    for f in frames:
        f = cv2.GaussianBlur(f, (3, 3), 0)
        noise = np.random.normal(0, 2, f.shape).astype(np.int16)
        f = np.clip(f.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        new_frames.append(f)
    return new_frames, fps

def aug_rotate(frames, fps, w, h):
    angle = random.uniform(*ROTATE_ANGLE_RANGE)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return [cv2.warpAffine(f, M, (w, h)) for f in frames], fps

# ===================== 二、时间增强（独立） =====================
def aug_time_scale(frames, fps, w, h):
    ratio = random.choice(SPEED_RATIOS)
    new_len = max(10, int(len(frames) * ratio))
    indices = np.linspace(0, len(frames)-1, new_len).astype(int)
    return [frames[i] for i in indices], fps * ratio

def aug_swap_frames(frames, fps, w, h):
    new_frames = frames.copy()
    for i in range(len(new_frames)-1):
        if random.random() < FRAME_SWAP_PROB:
            new_frames[i], new_frames[i+1] = new_frames[i+1], new_frames[i]
    return new_frames, fps

def aug_drop_interpolate(frames, fps, w, h):
    kept = [f for f in frames if random.random() > FRAME_DROP_PROB]
    target_len = len(frames)
    indices = np.linspace(0, len(kept)-1, target_len).astype(int)
    return [kept[i] for i in indices], fps

# ===================== 三、视频混合增强（独立） =====================
def aug_concat(frames, fps, w, h, all_videos):
    if len(all_videos) < 2:
        return frames, fps
    other_path = random.choice(all_videos)
    other_frames, _, _, _ = load_video(other_path)
    if len(other_frames) >= CONCAT_FRAME_COUNT:
        frames = frames + other_frames[:CONCAT_FRAME_COUNT]
    return frames, fps

# ===================== 四、遮挡增强（独立） =====================
def aug_random_erase(frames, fps, w, h):
    new_frames = []
    for f in frames:
        h_f, w_f = f.shape[:2]
        eh = random.randint(h_f//10, h_f//4)
        ew = random.randint(w_f//10, w_f//4)
        x = random.randint(0, w_f - ew)
        y = random.randint(0, h_f - eh)
        f[y:y+eh, x:x+ew] = np.random.randint(0,255,(eh,ew,3),dtype=np.uint8)
        new_frames.append(f)
    return new_frames, fps

def aug_edge_mask(frames, fps, w, h):
    new_frames = []
    mask_w = random.randint(10, 40)
    for f in frames:
        f[:, :mask_w] = 0
        f[:, -mask_w:] = 0
        new_frames.append(f)
    return new_frames, fps

def aug_local_mask(frames, fps, w, h):
    new_frames = []
    for f in frames:
        h_f = f.shape[0]
        y1, y2 = h_f//3, h_f//2
        f[y1:y2, :] = np.random.randint(0,255,(y2-y1, f.shape[1],3),dtype=np.uint8)
        new_frames.append(f)
    return new_frames, fps

# ===================== 增强列表（独立不叠加） =====================
AUG_LIST = [
    ("flip", aug_flip),
    # ("crop", aug_crop_scale),
    ("color", aug_color_jitter),
    ("blur_noise", aug_blur_noise),
    ("rotate", aug_rotate),
    ("speed", aug_time_scale),
    ("swap", aug_swap_frames),
    # ("drop", aug_drop_interpolate),
    # ("erase", aug_random_erase),
    # ("edge", aug_edge_mask),
    # ("local_mask", aug_local_mask),
]

# ===================== 批量处理入口 =====================
def process_single_video(video_path, save_dir, all_videos):
    name = Path(video_path).stem
    frames, fps, w, h = load_video(video_path)

    # 1. 基础独立增强
    for suffix, func in AUG_LIST:
        try:
            new_frames, new_fps = func(frames.copy(), fps, w, h)
            out_path = os.path.join(save_dir, f"{name}_{suffix}.mp4")
            save_video(new_frames, new_fps, w, h, out_path)
        except:
            continue

    # 2. 单独处理拼接增强
    # try:
    #     new_frames, new_fps = aug_concat(frames.copy(), fps, w, h, all_videos)
    #     out_path = os.path.join(save_dir, f"{name}_concat.mp4")
    #     save_video(new_frames, new_fps, w, h, out_path)
    # except:
    #     pass

# ===================== 主运行 =====================
if __name__ == "__main__":
    RAW_FOLDER = r"lstm_runs_fine/data/cup/0"
    SAVE_FOLDER = r"lstm_runs_fine/data/cup/0_aug"
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # 读取所有视频
    exts = ["*.mp4", "*.avi", "*.mov", "*.MP4", "*.AVI"]
    video_paths = []
    for ext in exts:
        video_paths += list(Path(RAW_FOLDER).rglob(ext))

    print(f"找到 {len(video_paths)} 个原始视频，开始独立增强...")

    for vp in tqdm(video_paths):
        process_single_video(str(vp), SAVE_FOLDER, [str(x) for x in video_paths])

    print("✅ 所有独立增强已完成！每个增强单独保存，互不叠加。")