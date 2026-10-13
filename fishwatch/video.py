import csv
import hashlib
import time
from pathlib import Path
import cv2
import numpy as np
from .io import write_json
from .features import inside, summarize_window, add_causal_baselines
from .tracking import Tracker, Detection, MotionDetector, YoloDetector, appearance


TRACK_FIELDS = ["frame", "time_s", "track_id", "cx", "cy", "x1", "y1", "x2", "y2", "confidence", "associated"]


class CSVDetector:
    """Streaming detections at native frame numbers; sorted input, bounded memory."""
    def __init__(self, path):
        self.file = Path(path).open(newline="", encoding="utf-8-sig")
        self.reader = csv.DictReader(self.file)
        required = {"frame", "x1", "y1", "x2", "y2", "confidence"}
        if not required <= set(self.reader.fieldnames or []):
            self.file.close()
            raise ValueError(f"检测CSV缺少列: {sorted(required-set(self.reader.fieldnames or []))}")
        self.last_frame = -1
        self.pending = self._next()

    def _next(self):
        row = next(self.reader, None)
        if row is not None:
            number = int(row["frame"])
            values = np.array([float(row[k]) for k in ("x1", "y1", "x2", "y2", "confidence")])
            if number < self.last_frame or number < 0 or not np.isfinite(values).all():
                raise ValueError("检测CSV必须按非负frame排序，且数值有限")
            if values[0] >= values[2] or values[1] >= values[3] or not 0 <= values[4] <= 1:
                raise ValueError("检测CSV包含非法检测框/置信度")
            self.last_frame = number
        return row

    def detect(self, frame, index):
        result = []
        while self.pending is not None and int(self.pending["frame"]) <= index:
            row = self.pending
            if int(row["frame"]) == index:
                box = np.array([float(row[k]) for k in ("x1", "y1", "x2", "y2")])
                box[[0, 2]] = np.clip(box[[0, 2]], 0, frame.shape[1])
                box[[1, 3]] = np.clip(box[[1, 3]], 0, frame.shape[0])
                if box[2] > box[0] and box[3] > box[1]:
                    result.append(Detection(box, float(row["confidence"]), appearance(frame, box)))
            self.pending = self._next()
        return result

    def close(self):
        self.file.close()


def analyze(video, output, cfg, detector="csv", detections=None, weights=None,
            class_id=0, domain="real", overlay=False, progress=True):
    video, output = Path(video).resolve(), Path(output).resolve()
    if not video.is_file():
        raise ValueError(f"视频不存在: {video}")
    output.mkdir(parents=True, exist_ok=True)
    if (output/"analysis.json").exists():
        raise ValueError(f"输出目录已含分析结果，请换目录以免覆盖: {output}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"无法解码视频: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
        cap.release()
        raise ValueError("视频帧率或尺寸无效，请转换为恒定帧率视频")
    step = max(1, int(round(fps/cfg["sample_fps"])))
    sample_dt = step/fps
    if cfg["window_seconds"] < 3*sample_dt:
        cap.release()
        raise ValueError("每窗口至少需要3个采样帧")
    if detector == "csv":
        if not detections:
            cap.release()
            raise ValueError("CSV模式需要 --detections")
        detect = CSVDetector(detections)
    elif detector == "yolo":
        detect = YoloDetector(weights or "", cfg, class_id)
    elif detector == "motion":
        detect = MotionDetector(cfg)
    else:
        raise ValueError("未知检测器")
    tracker = Tracker(cfg)
    windows, bucket, current_bucket = [], [], 0
    last_time, index, sampled = 0.0, 0, 0
    timing_fallback = False
    start_clock = time.monotonic()
    writer = None
    if overlay:
        writer = cv2.VideoWriter(str(output/"tracks.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 1/sample_dt, (width, height))
        if not writer.isOpened():
            raise ValueError("无法创建标注视频")
    try:
        with (output/"tracks.csv").open("w", newline="", encoding="utf-8") as file:
            csvout = csv.DictWriter(file, fieldnames=TRACK_FIELDS)
            csvout.writeheader()
            while cap.grab():
                if index % step:
                    index += 1
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError(f"第{index}帧解码失败，分析未完成")
                # Use actual decoder presentation timestamps when available; flag fallback.
                position = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
                if not np.isfinite(position) or (sampled and position <= last_time):
                    position = index/fps
                    timing_fallback = True
                if sampled and position <= last_time:
                    raise ValueError("非单调视频时间戳，请先转换为恒定帧率")
                t = float(position)
                target_bucket = int(t/cfg["window_seconds"])
                while current_bucket < target_bucket:
                    windows.append(summarize_window(bucket, current_bucket*cfg["window_seconds"], (current_bucket+1)*cfg["window_seconds"], cfg, width, height, sample_dt))
                    current_bucket += 1
                    bucket = []
                dets = detect.detect(frame, index) if detector == "csv" else detect.detect(frame)
                dets = [d for d in dets if d.confidence >= cfg["min_detection_confidence"] and inside(d.center/np.array([width, height]), cfg["tank_roi"])]
                rows = []
                for tid, det, associated in tracker.update(dets, t):
                    row = dict(zip(TRACK_FIELDS, [index, t, tid, *det.center, *det.box, det.confidence, int(associated)]))
                    csvout.writerow(row)
                    rows.append(row)
                    if writer:
                        x1, y1, x2, y2 = det.box.astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (110, 220, 110), 1)
                        cv2.putText(frame, f"track {tid}", (x1, max(y1-4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 220, 110), 1)
                if not rows:
                    csvout.writerow({"frame": index, "time_s": t, "track_id": -1})
                if writer:
                    cv2.putText(frame, f"{t:.1f}s | {domain} | short tracklets", (10, height-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
                    writer.write(frame)
                if sampled == 0:
                    cv2.imwrite(str(output/"preview.jpg"), frame)
                bucket.append({"time_s": t, "fish": rows})
                sampled += 1
                last_time = t
                index += 1
                if progress and sampled % 1000 == 0:
                    print(f"已分析 {t/60:.1f} 分钟 / {sampled} 采样帧", flush=True)
        if not sampled:
            raise ValueError("视频没有可用帧")
        if bucket:
            windows.append(summarize_window(bucket, current_bucket*cfg["window_seconds"], last_time+sample_dt, cfg, width, height, sample_dt))
    finally:
        cap.release()
        if writer:
            writer.release()
        if isinstance(detect, CSVDetector):
            detect.close()
    add_causal_baselines(windows)
    warnings = [
        "活动比例阈值为工程参数；遮蔽区停留仅统计可见鱼，不将漏检解释为躲藏。短期基线仅使用过去有效窗口，不代表昼夜基线。",
        "轨迹ID仅为短时关联，不保证跨遮挡、跨视频身份一致。",
        "路径为二维投影；未实现吞食、鳃盖或追逐动作分类。",
        "上层停留不等于水面呼吸；行为指标不能独立确定病因。"
    ]
    if timing_fallback:
        warnings.append("部分解码时间戳不可用，使用frame/fps；变帧率文件请先转恒定帧率。")
    if cfg["expected_fish_count"] is None:
        warnings.append("未设置实际鱼数：覆盖率仅表示至少检出一条，不能验证全缸鱼是否可见。")
    if detector == "motion":
        warnings.append("运动前景模式会漏掉静止鱼并检出反光/水草，禁止进入状态模型。")
    if domain == "synthetic":
        warnings.insert(0, "合成演示：非真实鱼视频，不具有生物学验证意义。")
    digest = hashlib.sha256()
    with video.open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            digest.update(chunk)
    result = {
        "schema_version": 1, "domain": domain, "video": str(video),
        "recording_id": digest.hexdigest()[:16], "video_sha256": digest.hexdigest(),
        "detector": detector, "config": cfg, "width": width, "height": height,
        "native_fps": fps, "sample_fps": 1/sample_dt, "duration_s": last_time+sample_dt,
        "sampled_frames": sampled, "elapsed_s": time.monotonic()-start_clock,
        "motion_unit": "reference_body_lengths/s" if cfg["reference_body_length_px"] else "projected_bbox_widths/s",
        "timing_fallback": timing_fallback, "warnings": warnings, "windows": windows
    }
    write_json(output/"analysis.json", result)
    return result
