import csv
import json
import shutil
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QInputDialog,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

try:
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
except Exception as e:
    raise ImportError("请先安装 mediapipe，例如: pip install mediapipe==0.10.14") from e


# ================= 全局配置与路径 =================

# 基础目录，存放模型、配置文件和输出结果
# BASE_DIR = Path(r"D:\FDM\my-AI-SOP")
BASE_DIR = Path(r"")


# SOP步骤定义的时间线CSV模板
TIMELINE_CSV = BASE_DIR / "full_timeline_10videos_template.csv"
# YOLOv8 目标检测模型路径
YOLO_MODEL_PATH = BASE_DIR / "yolov5/yolov5m.pt"
# LSTM 动作分类模型路径
LSTM_MODEL_PATH = BASE_DIR / "lstm/weight/best_model.pth"
# LSTM 模型配置文件路径
LSTM_CONFIG_PATH = BASE_DIR / "lstm/config.json"
# GUI运行结果输出目录
RUNS_GUI_DIR = BASE_DIR / "runs_gui"
# UI背景图片路径
UI_BG_PATH = BASE_DIR / "背景图片.jpg"

# YOLO检测类别名称
CLASS_NAMES = ["cup"]
# LSTM输入的多尺度时间窗口大小
# SOURCE_WINDOWS = [48, 72, 96]
SOURCE_WINDOWS = [72, 96, 128]

# S1步骤切换时的置信度容差 margin
S1_SWITCH_MARGIN = 0.05
# S1步骤切换的最小置信度阈值
S1_SWITCH_MIN_CONF = 0.6

#确认动作完成所需的连续命中帧数
CONFIRM_FRAMES_FIXED = 30
# YOLO检测框线条粗细
YOLO_BOX_THICKNESS = 3
# YOLO标签文字大小
YOLO_TEXT_SIZE = 38
# YOLO标签文字垂直偏移量
YOLO_TEXT_OFFSET_Y = 36
# 手部关键点线条粗细
HAND_LANDMARK_THICKNESS = 4
# 手部连接线粗细
HAND_CONNECTION_THICKNESS = 4
# 手部关键点圆圈半径
HAND_CIRCLE_RADIUS = 5
# 单个步骤最小持续时间（秒），防止瞬间误判
STEP_MIN_STAGE_SEC = 2.0

# 默认动作定义列表：(显示名称, 内部标签)
DEFAULT_ACTION_DEFS = [
    ("第一步：取出水杯", "S1_get_cup"),
    ("第二步：放入水杯", "S2_put_cup")
]

# 内部标签到中文显示名称的映射
ACTION_CN_MAP = {
    "S1_get_cup": "取出水杯",
    "S2_put_cup": "放入水杯",
    "background": "背景",
}

# YOLO类别ID/名称到中文名称的映射
YOLO_CN_MAP = {
    "cup": "水杯",
}


# ================= 数据类定义 =================

@dataclass
class RuntimeParams:
    """运行时参数配置"""
    yolo_conf: float          # YOLO检测置信度阈值
    lstm_conf: float          # LSTM分类置信度阈值
    confirm_frames: int       # 确认动作完成的连续帧数
    show_boxes: bool          # 是否显示YOLO检测框
    show_keypoints: bool      # 是否显示MediaPipe手部关键点
    save_snapshots: bool      # 是否保存动作完成时的截图
    save_log: bool            # 是否保存日志文件


@dataclass
class ActionRuntime:
    """动作运行时状态记录"""
    index: int                # 动作索引
    show: str                 # 显示名称
    fine_label: str           # 内部精细标签
    done: bool = False        # 是否已完成
    hit_count: int = 0        # 当前连续命中计数
    expected_conf: float = 0.0 # 当前预期置信度
    done_time_sec: Optional[float] = None # 完成时间（视频秒数）
    snapshot_path: Optional[str] = None   # 截图保存路径


# ================= 模型定义 =================

class ActionLSTM(torch.nn.Module):
    """
    基于LSTM的动作分类模型
    输入序列特征，输出动作类别概率
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: [batch, seq_len, input_size]
        out, _ = self.lstm(x)
        # 取最后一个时间步的输出进行分类
        return self.fc(out[:, -1, :])


# ================= 辅助函数 =================

def fine_label_from_row(step_id, _target_id, _event_type):
    """根据步骤ID映射到内部精细标签"""
    s = str(step_id)
    if s == "S1":
        return "S1_get_cup"
    if s == "S2":
        return "S2_put_cup"
    return "background"


def build_feature_row(detections, hands, frame_w, frame_h):
    """
    构建单帧的特征向量
    包含：YOLO检测结果特征 + MediaPipe手部关键点特征
    """
    # 初始化检测特征字典，每个类别保留5个特征: [conf, cx_norm, cy_norm, w_norm, h_norm]
    det_feat = {k: [0.0, 0.0, 0.0, 0.0, 0.0] for k in CLASS_NAMES}
    
    # 处理YOLO检测结果，只保留每个类别中置信度最高的那个检测框
    for d in detections:
        cls_name = d.get("cls_name")
        if cls_name not in det_feat:
            continue
        conf = float(d.get("conf", 0.0))
        x1, y1, x2, y2 = d.get("xyxy", [0, 0, 0, 0])
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        cx = float(x1) + w / 2.0
        cy = float(y1) + h / 2.0

        fw = max(float(frame_w), 1.0)
        fh = max(float(frame_h), 1.0)
        # 如果当前检测框置信度更高，则更新特征
        if conf > det_feat[cls_name][0]:
            det_feat[cls_name] = [conf, cx / fw, cy / fh, w / fw, h / fh]

    # 展平检测特征向量
    det_vec = []
    for cname in CLASS_NAMES:
        det_vec.extend(det_feat[cname])

    # 初始化手部特征向量 (2只手 * 21个点 * 3坐标 xyz)
    hand_vec = [0.0] * (2 * 21 * 3)
    for hi, hand in enumerate(hands[:2]):
        lms = hand.get("landmarks", [])
        for li, lm in enumerate(lms[:21]):
            base = hi * 63 + li * 3
            hand_vec[base] = float(lm[0]) if len(lm) > 0 else 0.0
            hand_vec[base + 1] = float(lm[1]) if len(lm) > 1 else 0.0
            hand_vec[base + 2] = float(lm[2]) if len(lm) > 2 else 0.0

    # 拼接检测特征和手部特征
    return np.array(det_vec + hand_vec, dtype=np.float32)


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    """将OpenCV BGR图像转换为Qt QImage用于显示"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    bytes_per_line = c * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


def to_beijing_time_str() -> str:
    """获取当前北京时间字符串"""
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")


def action_to_cn(label: str) -> str:
    """将内部动作标签转换为中文显示名称"""
    return ACTION_CN_MAP.get(label, label)


def _load_font(font_size: int = YOLO_TEXT_SIZE) -> ImageFont.FreeTypeFont:
    """加载支持中文的字体文件"""
    for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
        try:
            return ImageFont.truetype(fp, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_yolo_boxes_cn(frame: np.ndarray, boxes, names_map: dict):
    """
    在帧上绘制YOLO检测框和中文标签
    """
    vis = frame.copy()
    if boxes is None:
        return vis

    # 使用PIL进行中文文字绘制
    pil_img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(YOLO_TEXT_SIZE)

    for box in boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_name = names_map.get(cls_id, str(cls_id))
        cls_cn = YOLO_CN_MAP.get(cls_name, cls_name)
        
        # 绘制矩形框
        draw.rectangle([x1, y1, x2, y2], outline=(60, 220, 120), width=YOLO_BOX_THICKNESS)
        # 绘制中文标签
        draw.text((x1, max(2, y1 - YOLO_TEXT_OFFSET_Y)), cls_cn, fill=(60, 220, 120), font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ================= 核心推理线程 =================

class InferenceWorker(QThread):
    """
    后台推理线程
    负责加载模型、读取视频、执行YOLO检测和LSTM分类，并通过信号更新UI
    """
    sig_frame = pyqtSignal(QImage)      # 发送处理后的视频帧
    sig_status = pyqtSignal(dict)       # 发送状态信息（进度、置信度等）
    sig_action = pyqtSignal(dict)       # 发送动作完成事件
    sig_finished = pyqtSignal(dict)     # 发送任务完成信号
    sig_error = pyqtSignal(str)         # 发送错误信息

    def __init__(self, video_path: str, camera_index: int,params: RuntimeParams, parent =None):
        super().__init__(parent)
        self.video_path = Path(video_path)
        self.camera_index = camera_index
        self.params = params
        self._stop = False
        self._pause = False

        # 设置计算设备
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.id2label: Dict[int, str] = {}
        self.label2id: Dict[str, int] = {}

        # 模型实例
        self.lstm = None
        self.yolo = None
        self.hands = None

        # 动作状态管理
        self.actions: List[ActionRuntime] = []
        self.expected_idx = 0           # 当前期望进行的动作索引
        self.current_hit = 0            # 当前动作连续命中计数
        self.expected_stage_start_sec = 0.0 # 当前阶段开始时间
        self.last_expected_idx = -1     # 上一次期望的动作索引
        self.cpl_num = 0                   # 完成次数

        # 结果存储
        self.run_dir: Optional[Path] = None
        self.events: List[dict] = []

    def stop(self):
        """停止线程"""
        self._stop = True

    def toggle_pause(self):
        """切换暂停/继续状态"""
        self._pause = not self._pause

    def _load_models(self):
        """加载LSTM和YOLO模型及MediaPipe手部检测器"""
        # 加载LSTM配置以获取标签映射网络结构参数
        config = json.loads(LSTM_CONFIG_PATH.read_text(encoding="utf-8"))
        self.id2label = {int(x["id"]): x["label"] for x in config["label_map"]}
        self.label2id = {v: k for k, v in self.id2label.items()}

        # 初始化并加载LSTM模型权重
        self.lstm = ActionLSTM(
            config["input_size"],
            config["hidden_size"],
            config["num_layers"],
            config["num_classes"],
        ).to(self.device)
        self.lstm.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=self.device))
        self.lstm.eval()

        # 加载YOLO模型
        # self.yolo = YOLO(str(YOLO_MODEL_PATH))
        self.yolo = torch.hub.load('yolov5', 'custom', path=YOLO_MODEL_PATH, source='local').to(self.device)
        self.yolo.conf = self.params.yolo_conf
        self.yolo.classes = [41]  # 你指定的类别
        # 初始化MediaPipe手部检测器
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _build_actions(self):
        """
        构建动作列表
        优先从CSV时间线模板加载，如果失败或不存在则使用默认定义
        """
        self.actions = []

        if TIMELINE_CSV.exists():
            try:
                df = pd.read_csv(TIMELINE_CSV)
                video_name = self.video_path.name
                # 筛选当前视频对应的步骤并排序
                sdf = df[df["video_name"] == video_name].copy().sort_values(["start_sec", "end_sec"])
                if not sdf.empty:
                    for i, r in sdf.reset_index(drop=True).iterrows():
                        fine = fine_label_from_row(r["step_id"], r.get("target_id", ""), r.get("event_type", ""))
                        show_cn = ACTION_CN_MAP.get(fine, fine)
                        show = f"第{i + 1}步：{show_cn}"

                        self.actions.append(
                            ActionRuntime(
                                index=i,
                                show=show,
                                fine_label=fine,
                            )
                        )
            except Exception:
                self.actions = []

        # 如果没有从CSV加载到动作，使用默认定义
        if not self.actions:
            for i, (show, fine) in enumerate(DEFAULT_ACTION_DEFS):
                self.actions.append(ActionRuntime(index=i, show=show, fine_label=fine))

    def _init_run_dir(self):
        """初始化本次运行的输出目录"""
        RUNS_GUI_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = RUNS_GUI_DIR / f"{self.video_path.stem}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "snapshots").mkdir(exist_ok=True)

    def _save_snapshot(self, frame_bgr: np.ndarray, action: ActionRuntime, t_sec: float) -> str:
        """保存动作完成时的截图"""
        assert self.run_dir is not None
        fname = f"{action.index + 1:02d}_{action.fine_label}_{t_sec:.2f}.jpg".replace("/", "_")
        out = self.run_dir / "snapshots" / fname
        cv2.imwrite(str(out), frame_bgr)
        return str(out)

    def _calc_expected_conf(self, probs: torch.Tensor, expected_label: str) -> float:
        """计算当前帧中预期动作类别的置信度"""
        idx = self.label2id.get(expected_label)
        if idx is None:
            return 0.0
        return float(probs[0, idx].item())

    def _calc_s1_sibling_conf(self, _probs: torch.Tensor, _expected_label: str) -> float:
        # 当前场景没有S1双孔位，直接返回0
        return 0.0

    def _apply_expected_rules(self, _expected_action: ActionRuntime, expected_conf: float, _sibling_conf: float, t_sec: float) -> float:
        """
        应用业务规则过滤置信度
        例如：如果当前阶段持续时间太短，则忽略高置信度，防止快速切换导致的误判
        """
        stage_elapsed = t_sec - self.expected_stage_start_sec
        if stage_elapsed < STEP_MIN_STAGE_SEC:
            return 0.0

        return expected_conf

    def run(self):
        """
        线程主执行函数
        循环读取视频帧，进行推理，更新状态
        """
        try:
            self._load_models()
            self._build_actions()
            self._init_run_dir()

            # 发送初始动作定义给UI
            self.sig_status.emit({"action_defs": [a.show for a in self.actions], "total": len(self.actions)})

            if self.video_path is None and self.camera_index is None:
                raise RuntimeError("请先选择视频源")

            cap_path = self.camera_index if self.camera_index is not None else str(self.video_path)
            cap = cv2.VideoCapture(cap_path)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频: {cap_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_id = 0
            
            # 特征队列，用于存储历史帧特征以供LSTM使用
            feat_queue = deque(maxlen=max(SOURCE_WINDOWS))
            # 预测历史队列，用于平滑预测结果
            pred_history = deque(maxlen=5)

            current_pred = "background"

            while not self._stop:
                # 处理暂停状态
                if self._pause:
                    self.msleep(30)
                    continue

                ok, frame = cap.read()
                if not ok:
                    break

                t_sec = frame_id / fps
                frame_h, frame_w = frame.shape[:2]

                # 1. YOLO 目标检测
                yolo_result = self.yolo(frame)
                res = type('obj', (object,), {'boxes': [type('obj', (object,), {'xyxy': d[:4].unsqueeze(0), 'conf': d[4:5], 'cls': d[5:6]})() for d in yolo_result.xyxy[0]]})()

                # 如果需要，绘制检测框
                if self.params.show_boxes:
                    vis = draw_yolo_boxes_cn(frame, res.boxes, self.yolo.names)
                else:
                    vis = frame.copy()

                # 解析YOLO检测结果
                detections = []
                if res.boxes is not None:
                    for box in res.boxes:
                        cls_id = int(box.cls[0])
                        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                        detections.append(
                            {
                                "cls_name": self.yolo.names.get(cls_id, str(cls_id)),
                                "conf": float(box.conf[0]),
                                "xyxy": [x1, y1, x2, y2],
                            }
                        )

                # 2. MediaPipe 手部关键点检测
                rgb_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hr = self.hands.process(rgb_raw)
                hands_out = []
                if hr.multi_hand_landmarks:
                    for hidx, hand_lm in enumerate(hr.multi_hand_landmarks):
                        # 如果需要，绘制手部关键点
                        if self.params.show_keypoints:
                            mp_draw.draw_landmarks(
                                vis,
                                hand_lm,
                                mp_hands.HAND_CONNECTIONS,
                                mp_draw.DrawingSpec(color=(0, 255, 140), thickness=HAND_LANDMARK_THICKNESS, circle_radius=HAND_CIRCLE_RADIUS),
                                mp_draw.DrawingSpec(color=(0, 220, 120), thickness=HAND_CONNECTION_THICKNESS, circle_radius=HAND_CIRCLE_RADIUS),
                            )
                        lms = [[float(lm.x), float(lm.y), float(lm.z)] for lm in hand_lm.landmark]
                        hands_out.append({"hand_index": hidx, "landmarks": lms})

                # 3. 构建特征并入队
                feat_queue.append(build_feature_row(detections, hands_out, frame_w, frame_h))

                expected_show = "DONE"
                expected_conf = 0.0
                top3_text = "-"

                # 获取当前期望的动作
                if self.expected_idx < len(self.actions):
                    expected_action = self.actions[self.expected_idx]
                    expected_show = expected_action.show
                else:
                    expected_action = None

                # 如果期望动作索引发生变化，重置阶段开始时间
                if expected_action is not None and self.expected_idx != self.last_expected_idx:
                    self.expected_stage_start_sec = t_sec
                    self.last_expected_idx = self.expected_idx

                # 4. LSTM 动作分类推理
                # 只有当特征队列长度足够时才进行推理
                if len(feat_queue) >= min(SOURCE_WINDOWS):
                    feat_arr = np.stack(feat_queue, axis=0)
                    scale_preds = []
                    expected_confs = []
                    sibling_confs = []
                    top3_candidates = []

                    # 多尺度窗口投票机制
                    for src_len in SOURCE_WINDOWS:
                        if len(feat_arr) < src_len:
                            continue

                        # 从队列中均匀采样指定长度的序列
                        src_seq = feat_arr[-src_len:]
                        pos = np.linspace(0, src_len - 1, 48)
                        idx = np.clip(np.round(pos).astype(int), 0, src_len - 1)
                        seq = src_seq[idx]

                        xb = torch.from_numpy(seq[None, ...]).float().to(self.device)
                        with torch.no_grad():
                            logits = self.lstm(xb)
                            probs = torch.softmax(logits, dim=1)

                        probs_np = probs[0].detach().cpu().numpy()
                        
                        # 收集Top3候选项用于显示
                        for i_cls, p_cls in enumerate(probs_np):
                            lb = self.id2label.get(i_cls, "")
                            if lb and lb != "background":
                                top3_candidates.append((lb, float(p_cls)))

                        # 获取当前尺度的预测结果
                        pid = int(torch.argmax(probs, dim=1).item())
                        scale_preds.append(self.id2label.get(pid, "background"))

                        # 计算预期动作的置信度
                        if expected_action is not None:
                            expected_confs.append(self._calc_expected_conf(probs, expected_action.fine_label))
                            sibling_confs.append(self._calc_s1_sibling_conf(probs, expected_action.fine_label))

                    # 聚合多尺度结果
                    if scale_preds:
                        # 处理Top3显示文本
                        if top3_candidates:
                            agg = {}
                            for lb, p in top3_candidates:
                                agg[lb] = max(agg.get(lb, 0.0), p)
                            top3_sorted = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:3]
                            top3_text = " | ".join([f"{action_to_cn(lb)}:{p:.2f}" for lb, p in top3_sorted])

                        # 投票决定当前帧的最终预测类别
                        votes = {}
                        for p in scale_preds:
                            votes[p] = votes.get(p, 0) + 1
                        current_pred = max(votes, key=votes.get)
                        
                        # 加入历史平滑队列
                        pred_history.append(current_pred)
                        smooth_votes = {}
                        for p in pred_history:
                            smooth_votes[p] = smooth_votes.get(p, 0) + 1
                        current_pred = max(smooth_votes, key=smooth_votes.get)

                        # 判断当前期望动作是否完成
                        if expected_action is not None:
                            expected_conf = float(np.mean(expected_confs)) if expected_confs else 0.0
                            sibling_conf = float(np.mean(sibling_confs)) if sibling_confs else 0.0
                            
                            # 应用业务规则过滤
                            expected_conf = self._apply_expected_rules(expected_action, expected_conf, sibling_conf, t_sec)

                            # 如果置信度超过阈值，增加命中计数
                            if expected_conf >= self.params.lstm_conf:
                                self.current_hit += 1
                            else:
                                self.current_hit = 0

                            expected_action.expected_conf = expected_conf
                            expected_action.hit_count = self.current_hit

                            # 如果连续命中帧数达到要求，标记动作完成
                            if self.current_hit >= self.params.confirm_frames:
                                expected_action.done = True
                                expected_action.done_time_sec = t_sec

                                # 保存截图
                                if self.params.save_snapshots:
                                    expected_action.snapshot_path = self._save_snapshot(vis, expected_action, t_sec)

                                done_time_beijing = to_beijing_time_str()

                                self.events.append(
                                    {
                                        "index": expected_action.index + 1,
                                        "action": expected_action.show,
                                        "fine_label": expected_action.fine_label,
                                        "done_time_sec": round(t_sec, 3),
                                        "done_time_beijing": done_time_beijing,
                                        "snapshot_path": expected_action.snapshot_path or "",
                                    }
                                )

                                # 发送动作完成信号给UI
                                self.sig_action.emit(
                                    {
                                        "index": expected_action.index,
                                        "status": "完成",
                                        "info": f"完成时间（北京时间）: {done_time_beijing}",
                                        "snapshot": expected_action.snapshot_path,
                                    }
                                )

                                # 移动到下一个动作
                                
                                self.expected_idx += 1
                                self.current_hit = 0

                                if self.expected_idx == len(self.actions):
                                    self.cpl_num+=1
                                    self.expected_idx=0

                # 发送当前帧的状态信息给UI
                status = {
                    "frame_id": frame_id,
                    "time_sec": t_sec,
                    "current_pred": action_to_cn(current_pred),
                    "expected_show": expected_show,
                    "expected_conf": expected_conf,
                    "lstm_conf": self.params.lstm_conf,
                    "hit": self.current_hit,
                    "confirm_frames": self.params.confirm_frames,
                    "top3": top3_text,
                    "progress": self.expected_idx,
                    "total": len(self.actions),
                    "cpl_num": self.cpl_num
                }
                self.sig_status.emit(status)
                self.sig_frame.emit(bgr_to_qimage(vis))

                frame_id += 1

            # 清理资源
            cap.release()
            if self.hands is not None:
                self.hands.close()

            # 保存日志
            result = {"run_dir": str(self.run_dir) if self.run_dir else "", "events": self.events}
            if self.params.save_log and self.run_dir:
                self._save_logs()

            self.sig_finished.emit(result)

        except Exception as e:
            self.sig_error.emit(str(e))

    def _save_logs(self):
        """保存JSON和CSV格式的结果日志"""
        assert self.run_dir is not None
        json_path = self.run_dir / "result.json"
        csv_path = self.run_dir / "events.csv"

        result_obj = {
            "video": self.video_path.name,
            "completed": sum(1 for a in self.actions if a.done),
            "total": len(self.actions),
            "actions": [
                {
                    "index": a.index + 1,
                    "show": a.show,
                    "fine_label": a.fine_label,
                    "done": a.done,
                    "done_time_sec": a.done_time_sec,
                    "done_time_beijing": None,
                    "snapshot_path": a.snapshot_path,
                }
                for a in self.actions
            ],
        }
        json_path.write_text(json.dumps(result_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["index", "action", "fine_label", "done_time_sec", "done_time_beijing", "snapshot_path"],
            )
            writer.writeheader()
            for row in self.events:
                writer.writerow(row)


# ================= UI 组件 =================

class AspectRatioLabel(QLabel):
    """保持固定宽高比的标签控件，用于显示视频或截图"""
    def __init__(self, text="", ratio_w=16, ratio_h=9, parent=None):
        super().__init__(text, parent)
        self.ratio_w = ratio_w
        self.ratio_h = ratio_h
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_ratio(self, ratio_w: int, ratio_h: int):
        if ratio_w <= 0 or ratio_h <= 0:
            return
        self.ratio_w = ratio_w
        self.ratio_h = ratio_h
        self.updateGeometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = max(1, self.width())
        h = int(w * self.ratio_h / self.ratio_w)
        self.setFixedHeight(h)


class ActionCard(QFrame):
    """
    动作卡片组件
    显示单个动作的名称、状态、截图和元数据
    """
    def __init__(self, idx: int, name: str, parent=None):
        super().__init__(parent)
        self.idx = idx
        self.name = name

        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(190, 185)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.lbl_title = QLabel(name)
        self.lbl_status = QLabel("状态: 未开始")
        self.lbl_image = AspectRatioLabel("截图", 16, 9)
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_meta = QLabel("--")

        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.lbl_image)
        layout.addWidget(self.lbl_meta)

    def apply_style(self, alpha: int):
        """应用半透明样式"""
        a = max(0, min(100, int(alpha)))
        card_bg = int(a * 1.6)
        self.setStyleSheet(
            f"QFrame {{ background: rgba(8,22,44,{card_bg}); border: 1px solid rgba(75,170,255,0.68); border-radius: 12px; }}"
        )
        self.lbl_title.setStyleSheet("color: #d7f2ff; font-weight: 700; font-size: 20px;")
        self.lbl_status.setStyleSheet("color: #bde8ff; font-size: 16px;")
        self.lbl_image.setStyleSheet(
            f"background: rgba(4,14,28,{max(20, card_bg-30)}); color: #9fdfff; border: 1px dashed rgba(90,190,255,0.68); border-radius: 8px;"
        )
        self.lbl_meta.setStyleSheet("color: #9fdfff; font-size: 15px;")

    def set_status(self, status: str, info: str = "--"):
        """设置状态文本"""
        self.lbl_status.setText(f"状态: {status}")
        self.lbl_meta.setText(info)

    def set_snapshot(self, snapshot_path: Optional[str]):
        """设置并显示截图"""
        if not snapshot_path:
            return
        p = Path(snapshot_path)
        if not p.exists():
            return
        pix = QPixmap(str(p))
        if pix.isNull():
            return
        scaled = pix.scaled(self.lbl_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_image.setPixmap(scaled)
        self.lbl_image.setText("")


# ================= 主窗口 =================

class MainWindow(QMainWindow):
    """
    主界面窗口
    包含视频显示区、参数控制区、动作卡片展示区
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-SOP 动作识别系统")
        self.resize(1520, 920)
        self.video_path: Optional[Path] = None
        self.camera_index: Optional[int] = None
        self.worker: Optional[InferenceWorker] = None
        self.last_run_dir: Optional[Path] = None
        self.panel_alpha = 72

        self.cards: List[ActionCard] = []
        self._apply_theme()
        self._build_ui()

    def _apply_theme(self):
        """应用全局QSS样式表"""
        bg_path = UI_BG_PATH.as_posix() if UI_BG_PATH.exists() else ""
        a = max(0, min(100, int(self.panel_alpha)))
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background: #020812;
            }}
            QWidget {{ background: transparent; color: #dff5ff; font-family: 'Microsoft YaHei'; font-size: 13px; }}
            QWidget#root_bg {{
                border-image: url('{bg_path}') 0 0 0 0 stretch stretch;
            }}
            QGroupBox {{ border: 1px solid rgba(60,160,240,0.78); border-radius: 12px; margin-top: 12px; padding-top: 12px; background-color: rgba(4,16,32,{a}); }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 8px; color: #9fe6ff; font-weight: 700; }}
            QLabel {{ color: #dff5ff; background: transparent; }}
            QLineEdit, QSpinBox {{ background-color: rgba(7,23,45,{max(20, a)}); border: 1px solid rgba(90,190,255,0.85); border-radius: 8px; padding: 6px; color: #eaf9ff; }}
            QProgressBar {{ border: 1px solid rgba(90,190,255,0.85); border-radius: 8px; text-align: center; background: rgba(7,23,45,{max(18, a-10)}); color: #e6f8ff; min-height: 18px; }}
            QProgressBar::chunk {{ background-color: rgba(0,224,255,0.95); border-radius: 8px; }}
            QPushButton {{ background-color: rgba(8,26,52,{max(24, a+4)}); border: 1px solid rgba(90,190,255,0.95); border-radius: 10px; padding: 7px 12px; color: #e8fbff; font-weight: 700; }}
            QPushButton:hover {{ background-color: rgba(12,40,80,{max(30, a+10)}); }}
            QPushButton:pressed {{ background-color: rgba(6,18,38,{max(20, a-8)}); }}
            QScrollArea {{ border: 1px solid rgba(60,160,240,0.72); border-radius: 10px; background: rgba(4,16,32,{max(16, a-16)}); }}
            QSlider::groove:horizontal {{ height: 6px; background: rgba(145,222,255,0.28); border-radius: 3px; }}
            QSlider::handle:horizontal {{ width: 14px; margin: -4px 0; border-radius: 7px; background: #00e6ff; border: 1px solid #9defff; }}
            """
        )

    def _build_ui(self):
        """构建用户界面布局"""
        root = QWidget()
        root.setObjectName("root_bg")
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # --- 顶部栏 ---
        top_bar = QFrame()
        top_bar.setStyleSheet("QFrame{background:rgba(4,16,32,0.95);border:2px solid rgba(80,180,255,0.98);border-radius:18px;}")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(18, 14, 18, 14)
        top_layout.setSpacing(14)

        lbl_title = QLabel("AI-SOP 动作识别系统")
        lbl_title.setFont(QFont("Microsoft YaHei", 28, QFont.Bold))
        lbl_title.setStyleSheet("color:#baf2ff; padding: 10px 16px; border:1px solid rgba(80,180,255,0.98); border-radius:14px; background:rgba(8,30,56,0.82);")

        self.lbl_video = QLabel("视频: 未加载")
        self.lbl_video.setFont(QFont("Microsoft YaHei", 17, QFont.Bold))

        self.end_step=len(DEFAULT_ACTION_DEFS)
        self.lbl_step = QLabel(f"当前步骤: 00/{self.end_step:02d}")
        self.lbl_step.setFont(QFont("Microsoft YaHei", 17, QFont.Bold))
        
        self.cpl_num = QLabel("完成次数: 00")
        self.cpl_num.setFont(QFont("Microsoft YaHei", 17, QFont.Bold))
        self.progress = QProgressBar()
        self.progress.setMaximum(5)
        self.progress.setValue(0)
        self.progress.setFixedWidth(290)
        self.progress.setMinimumHeight(40)

        top_layout.addWidget(lbl_title)
        top_layout.addStretch(1)
        self.lbl_video.setStyleSheet("padding: 8px 14px; border:1px solid rgba(80,180,255,0.95); border-radius:12px; background:rgba(8,30,56,0.72);")
        self.lbl_step.setStyleSheet("padding: 8px 14px; border:1px solid rgba(80,180,255,0.95); border-radius:12px; background:rgba(8,30,56,0.72);")
        self.cpl_num.setStyleSheet("padding: 8px 14px; border:1px solid rgba(80,180,255,0.95); border-radius:12px; background:rgba(8,30,56,0.72);")

        top_layout.addWidget(self.lbl_video)
        top_layout.addWidget(self.lbl_step)
        top_layout.addWidget(self.cpl_num)
        top_layout.addWidget(self.progress)

        # --- 中间分割区 ---
        mid_split = QSplitter(Qt.Horizontal)
        mid_split.setChildrenCollapsible(False)

        # 左侧：视频显示和实时信息
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.video_label = QLabel("视频显示区")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: rgba(2,10,24,0.72); color: #8fdfff; border: 1px solid rgba(80,180,255,0.72); border-radius: 12px;")
        self.video_label.setMinimumHeight(420)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        frame_info = QGroupBox("")
        frame_info.setStyleSheet("QGroupBox{background: rgba(4,16,32,0.94); border:2px solid rgba(80,180,255,0.95); border-radius:14px; margin-top:8px; padding-top:8px;}")
        fi = QGridLayout(frame_info)
        fi.setContentsMargins(16, 12, 16, 14)
        fi.setHorizontalSpacing(20)
        fi.setVerticalSpacing(12)

        self.lbl_frameinfo_title = QLabel("当前帧信息")
        self.lbl_frameinfo_title.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        self.lbl_frameinfo_title.setStyleSheet("color:#bdf0ff; padding-bottom: 4px;")

        self.lbl_frame_val = QLabel("0")
        self.lbl_time_val = QLabel("0.00s")
        self.lbl_current_val = QLabel("-")
        self.lbl_expected_val = QLabel("-")
        self.lbl_conf_val = QLabel("0.000")
        self.lbl_hit_val = QLabel("0/5")
        self.lbl_top3_val = QLabel("-")
        self.lbl_top3_val.setWordWrap(True)

        info_font = QFont("Microsoft YaHei", 17)
        for w in [self.lbl_frame_val, self.lbl_time_val, self.lbl_current_val, self.lbl_expected_val, self.lbl_conf_val, self.lbl_hit_val, self.lbl_top3_val]:
            w.setFont(info_font)

        self.lbl_k_frame = QLabel("帧号:")
        self.lbl_k_time = QLabel("时间:")
        self.lbl_k_current = QLabel("当前动作:")
        self.lbl_k_expected = QLabel("预期动作:")
        self.lbl_k_conf = QLabel("预期置信度:")
        self.lbl_k_hit = QLabel("命中帧:")
        self.lbl_k_top3 = QLabel("Top3:")

        key_font = QFont("Microsoft YaHei", 16, QFont.Bold)
        for k in [self.lbl_k_frame, self.lbl_k_time, self.lbl_k_current, self.lbl_k_expected, self.lbl_k_conf, self.lbl_k_hit, self.lbl_k_top3]:
            k.setFont(key_font)
            k.setStyleSheet("color:#c7efff;")

        fi.addWidget(self.lbl_frameinfo_title, 0, 0, 1, 4)

        fi.addWidget(self.lbl_k_frame, 1, 0)
        fi.addWidget(self.lbl_frame_val, 1, 1)
        fi.addWidget(self.lbl_k_time, 1, 2)
        fi.addWidget(self.lbl_time_val, 1, 3)
        fi.addWidget(self.lbl_k_current, 2, 0)
        fi.addWidget(self.lbl_current_val, 2, 1, 1, 3)
        fi.addWidget(self.lbl_k_expected, 3, 0)
        fi.addWidget(self.lbl_expected_val, 3, 1, 1, 3)
        fi.addWidget(self.lbl_k_conf, 4, 0)
        fi.addWidget(self.lbl_conf_val, 4, 1)
        fi.addWidget(self.lbl_k_hit, 4, 2)
        fi.addWidget(self.lbl_hit_val, 4, 3)
        fi.addWidget(self.lbl_k_top3, 5, 0)
        fi.addWidget(self.lbl_top3_val, 5, 1, 1, 3)

        left_layout.addWidget(self.video_label, 3)
        left_layout.addWidget(frame_info, 2)

        # 右侧：参数控制和操作按钮
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        params = QGroupBox("参数设置")
        p = QGridLayout(params)
        self.edit_yolo = QLineEdit("0.30")
        self.edit_lstm = QLineEdit("0.50")
        self.edit_windows = QLineEdit("72,96,128")
        self.edit_timeout = QLineEdit("10.0")

        rows = [
            ("YOLO阈值", self.edit_yolo),
            ("LSTM阈值", self.edit_lstm),
            ("多尺度窗口", self.edit_windows),
            ("超时阈值(秒)", self.edit_timeout),
        ]
        for r, (name, w) in enumerate(rows):
            p.addWidget(QLabel(name), r, 0)
            p.addWidget(w, r, 1)

        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(self.panel_alpha)
        self.alpha_slider.valueChanged.connect(self.on_alpha_changed)
        self.lbl_alpha = QLabel(f"透明度: {self.panel_alpha}")
        p.addWidget(self.lbl_alpha, len(rows), 0)
        p.addWidget(self.alpha_slider, len(rows), 1)

        options = QGroupBox("显示选项")
        o = QVBoxLayout(options)
        self.chk_boxes = QCheckBox("显示检测框")
        self.chk_keypoints = QCheckBox("显示关键点")
        self.chk_snapshots = QCheckBox("保存截图")
        self.chk_log = QCheckBox("保存日志")
        for cb in [self.chk_boxes, self.chk_keypoints, self.chk_snapshots, self.chk_log]:
            cb.setChecked(True)
            o.addWidget(cb)

        controls = QGroupBox("操作控制")
        c = QVBoxLayout(controls)
        self.btn_import = QPushButton("导入视频")
        self.btn_start = QPushButton("开始分析")
        self.btn_pause = QPushButton("暂停/继续")
        self.btn_stop = QPushButton("停止")
        self.btn_camera = QPushButton("加载相机")
        self.btn_export = QPushButton("导出结果")

        row1 = QHBoxLayout(); row1.addWidget(self.btn_import); row1.addWidget(self.btn_camera)
        row2 = QHBoxLayout(); row2.addWidget(self.btn_start)
        row3 = QHBoxLayout(); row3.addWidget(self.btn_pause); row3.addWidget(self.btn_stop)
        row4 = QHBoxLayout(); row4.addWidget(self.btn_export)
        c.addLayout(row1); c.addLayout(row2); c.addLayout(row3);c.addLayout(row4)

        right_layout.addWidget(params)
        right_layout.addWidget(options)
        right_layout.addWidget(controls)
        right_layout.addStretch(1)

        mid_split.addWidget(left_panel)
        mid_split.addWidget(right_panel)
        mid_split.setStretchFactor(0, 3)
        mid_split.setStretchFactor(1, 1)

        # --- 底部动作卡片区 ---
        cards_container = QWidget()
        self.cards_layout = QGridLayout(cards_container)
        self.cards_layout.setSpacing(12)
        self.cards_layout.setContentsMargins(4, 4, 4, 4)

        for i in range(5):
            card = ActionCard(i, f"{i + 1:02d}. 动作")
            card.apply_style(self.panel_alpha)
            self.cards.append(card)
            self.cards_layout.addWidget(card, 0, i)

        cards_scroll = QScrollArea()
        cards_scroll.setWidgetResizable(True)
        cards_scroll.setWidget(cards_container)
        cards_scroll.setMinimumHeight(460)

        main_layout.addWidget(top_bar)
        main_layout.addWidget(mid_split, 3)
        main_layout.addWidget(cards_scroll, 3)

        # 绑定信号槽
        self.btn_import.clicked.connect(self.on_import_video)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_export.clicked.connect(self.on_export)
        self.btn_camera.clicked.connect(self.on_open_camera)

    def on_alpha_changed(self, v: int):
        """透明度滑块改变事件"""
        self.panel_alpha = int(v)
        if hasattr(self, "lbl_alpha"):
            self.lbl_alpha.setText(f"透明度: {self.panel_alpha}")
        self._apply_theme()
        for c in self.cards:
            c.apply_style(self.panel_alpha)


    def on_open_camera(self):
        """打开摄像头"""
        # 弹出选择摄像头索引（默认0为第一个摄像头）
        camera_index, ok = QInputDialog.getInt(self, "选择摄像头", "摄像头索引（通常0为默认摄像头）：", 0, 0, 10)
        if not ok:
            return

        # 打开摄像头
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            QMessageBox.warning(self, "打开失败", f"无法打开索引为 {camera_index} 的摄像头，请检查设备是否被占用。")
            return

        # 读取第一帧
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            QMessageBox.warning(self, "打开失败", "摄像头无法读取画面，请检查设备权限。")
            return

        h, w = frame.shape[:2]
        cap.release()

        # 更新UI：标记为摄像头模式
        self.camera_index = camera_index  # 用特殊路径标记是摄像头
        self.video_path = None
        self.lbl_video.setText(f"摄像头: 索引{camera_index} ({w}x{h})")

        # 设置预览图（同视频逻辑）
        pix = QPixmap.fromImage(bgr_to_qimage(frame))
        self._set_video_preview_pixmap(pix)

    def on_import_video(self):
        """导入视频文件"""
        f, _ = QFileDialog.getOpenFileName(self, "选择视频", str(BASE_DIR / "video"), "Video (*.mp4 *.avi *.mov *.mkv *.wmv)")
        if not f:
            return

        path = Path(f)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            QMessageBox.warning(self, "导入失败", "该视频无法打开，请检查编码格式或路径。")
            return

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            QMessageBox.warning(self, "导入失败", "该视频无法读取帧，请更换视频。")
            return

        h, w = frame.shape[:2]
        cap.release()

        self.video_path = path
        self.camera_index= None
        self.lbl_video.setText(f"视频: {self.video_path.name} ({w}x{h})")

        pix = QPixmap.fromImage(bgr_to_qimage(frame))
        self._set_video_preview_pixmap(pix)

    def _reset_cards(self, names: List[str]):
        """重置所有动作卡片的状态"""
        for i, card in enumerate(self.cards):
            if i < len(names):
                card.lbl_title.setText(names[i])
                card.set_status("未开始", "--")
                card.lbl_image.setPixmap(QPixmap())
                card.lbl_image.setText("截图")
            else:
                card.lbl_title.setText(f"{i + 1:02d}. -")
                card.set_status("-", "--")

    def _params(self) -> RuntimeParams:
        """从UI控件获取当前运行时参数"""
        return RuntimeParams(
            yolo_conf=float(self.edit_yolo.text().strip()),
            lstm_conf=float(self.edit_lstm.text().strip()),
            confirm_frames=CONFIRM_FRAMES_FIXED,
            show_boxes=self.chk_boxes.isChecked(),
            show_keypoints=self.chk_keypoints.isChecked(),
            save_snapshots=self.chk_snapshots.isChecked(),
            save_log=self.chk_log.isChecked(),
        )

    def on_start(self):
        """开始分析按钮点击事件"""
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "提示", "正在分析中")
            return
        
        if self.video_path is None and self.camera_index is None:
            QMessageBox.warning(self, "提示", "请先选择视频源")
            return

        try:
            params = self._params()
        except Exception:
            QMessageBox.warning(self, "参数错误", "请检查阈值参数格式")
            return

        self.progress.setValue(0)
        self.lbl_step.setText(f"当前步骤: 00/{self.end_step:02d}")
        self.cpl_num.setText("完成次数: 00")
        # 创建并启动推理线程
        self.worker = InferenceWorker(str(self.video_path),self.camera_index, params)
        self.worker.sig_frame.connect(self.on_frame)
        self.worker.sig_status.connect(self.on_status)
        self.worker.sig_action.connect(self.on_action)
        self.worker.sig_finished.connect(self.on_finished)
        self.worker.sig_error.connect(self.on_error)
        self.worker.start()

    def on_pause(self):
        """暂停/继续按钮点击事件"""
        if self.worker and self.worker.isRunning():
            self.worker.toggle_pause()

    def on_stop(self):
        """停止按钮点击事件"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)

    def on_export(self):
        """导出结果按钮点击事件"""
        if not self.last_run_dir or not self.last_run_dir.exists():
            QMessageBox.information(self, "提示", "暂无可导出结果")
            return
        dst = QFileDialog.getExistingDirectory(self, "选择导出目录", str(BASE_DIR))
        if not dst:
            return
        out = Path(dst) / self.last_run_dir.name
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(self.last_run_dir, out)
        QMessageBox.information(self, "导出完成", f"已导出到: {out}")

    def _set_video_preview_pixmap(self, pix: QPixmap):
        """缩放并设置视频预览图片"""
        if pix.isNull():
            return
        scaled = pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)
        self.video_label.setText("")

    def resizeEvent(self, event):
        """窗口大小改变时重新缩放视频预览"""
        super().resizeEvent(event)
        pix = self.video_label.pixmap()
        if pix is not None and not pix.isNull():
            self._set_video_preview_pixmap(pix)

    def on_frame(self, qimg: QImage):
        """接收推理线程发送的视频帧并显示"""
        pix = QPixmap.fromImage(qimg)
        self._set_video_preview_pixmap(pix)

    def on_status(self, st: dict):
        """接收推理线程发送的状态信息并更新UI"""
        if "action_defs" in st:
            self._reset_cards(st["action_defs"])
            self.progress.setMaximum(st.get("total", 5))
            return

        self.lbl_frame_val.setText(str(st.get("frame_id", 0)))
        self.lbl_time_val.setText(f"{st.get('time_sec', 0.0):.2f}s")
        self.lbl_current_val.setText(st.get('current_pred', '-'))
        self.lbl_expected_val.setText(st.get("expected_show", "-"))
        self.lbl_conf_val.setText(f"{st.get('expected_conf', 0.0):.3f}")
        self.lbl_hit_val.setText(f"{st.get('hit', 0)}/{st.get('confirm_frames', 0)}")
        self.lbl_top3_val.setText(st.get('top3', '-'))

        p = st.get("progress", 0)
        total = st.get("total", 5)
        self.progress.setMaximum(total)
        self.progress.setValue(p)
        self.lbl_step.setText(f"当前步骤: {p:02d}/{total:02d}")
        self.cpl_num.setText(f"完成次数: {st.get('cpl_num', 00):.2f}")
        if p < len(self.cards):
            self.cards[p].set_status("进行中", "")

    def on_action(self, ev: dict):
        """接收动作完成事件并更新对应卡片"""
        idx = int(ev.get("index", -1))
        if 0 <= idx < len(self.cards):
            self.cards[idx].set_status(ev.get("status", "完成"), ev.get("info", "--"))
            self.cards[idx].set_snapshot(ev.get("snapshot"))

    def on_finished(self, result: dict):
        """推理完成回调"""
        run_dir = result.get("run_dir", "")
        self.last_run_dir = Path(run_dir) if run_dir else None
        QMessageBox.information(self, "完成", "视频分析完成")

    def on_error(self, msg: str):
        """错误处理回调"""
        print(f"Error: {msg}")
        QMessageBox.critical(self, "错误", msg)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()