"""Synthetic scenario generator. Its labels are invented, not fish physiology."""
import csv
from pathlib import Path
import cv2
import numpy as np
from .io import config_from, write_json
from .video import analyze
from .model import train, infer
from .report import render_report


SCENARIOS = [
    ("reference", "reference", "medium"),
    ("low", "reference", "low"),
    ("high", "reference", "high"),
    ("reference", "low", "low"),
    ("low", "low", "high"),
    ("high", "reference", "medium")
]


def synth_video(folder, scenario, seed, seconds=60, fps=8):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/"synthetic.mp4"
    width, height = 480, 270
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise ValueError("OpenCV无法创建演示视频")
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2*np.pi, 3)
    offsets = rng.uniform(-0.025, 0.025, 3)
    angles = phases.copy()
    colors = [(206, 222, 166), (120, 186, 238), (211, 158, 136)]
    baseline = ("reference", "reference", "medium")
    try:
        with (folder/"detections.csv").open("w", newline="", encoding="utf-8") as file:
            boxes = csv.DictWriter(file, fieldnames=["frame", "x1", "y1", "x2", "y2", "confidence"])
            boxes.writeheader()
            for frame_idx in range(int(seconds*fps)):
                t = frame_idx/fps
                state = scenario if seconds/3 <= t < 2*seconds/3 else baseline
                temp, oxygen, response = state
                # Deliberately artificial mappings used ONLY for software testing.
                rate = {"low": 0.35, "reference": 1.0, "high": 2.0}[temp]
                angles += rate*np.array([0.55, 0.63, 0.71])/fps
                frame = np.zeros((height, width, 3), np.uint8)
                frame[:] = (48, 49, 24)
                for y in range(height):
                    frame[y, :, 0] = 65-int(y/height*25)
                    frame[y, :, 1] = 65-int(y/height*30)
                cv2.line(frame, (0, 12), (width, 12), (145, 157, 107), 1)
                for j in range(3):
                    cx = {"low": 0.29, "medium": 0.50, "high": 0.73}[response] + 0.16*np.sin(angles[j])+offsets[j]
                    cy = (0.13+0.11*j if oxygen == "low" else 0.30+0.20*j)+0.035*np.sin(angles[j]*0.7)
                    x, y = int(cx*width), int(cy*height)
                    cv2.ellipse(frame, (x, y), (18, 26), 0, 0, 360, colors[j], -1)
                    tail = np.array([[x-17, y], [x-30, y-14], [x-30, y+14]])
                    cv2.fillPoly(frame, [tail], colors[j])
                    cv2.line(frame, (x, y-24), (x-7, y-35), colors[j], 2)
                    cv2.line(frame, (x, y+23), (x-8, y+35), colors[j], 2)
                    cv2.circle(frame, (x+10, y-6), 3, (30, 35, 20), -1)
                    cv2.line(frame, (x-4, y-20), (x-4, y+20), (60, 75, 40), 2)
                    boxes.writerow({"frame": frame_idx, "x1": x-20, "y1": max(y-28, 0), "x2": x+20, "y2": min(y+28, height), "confidence": 0.99})
                cv2.putText(frame, "SYNTHETIC DEMO - NOT REAL FISH", (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (195, 223, 205), 1)
                cv2.putText(frame, f"{t:.1f}s  seed:{seed}", (12, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 180, 160), 1)
                writer.write(frame)
    finally:
        writer.release()
    with (folder/"labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer_csv = csv.DictWriter(file, fieldnames=["start_s", "end_s", "task", "label", "source"])
        writer_csv.writeheader()
        for start, end, state in [(0, seconds/3, baseline), (seconds/3, 2*seconds/3, scenario), (2*seconds/3, seconds, baseline)]:
            for task, label in zip(["temperature", "oxygen", "feeding_response"], state):
                writer_csv.writerow({"start_s": start, "end_s": end, "task": task, "label": label, "source": "synthetic_generator_not_biological_truth"})
    return path


def demo(output, backend="logistic", epochs=40):
    output = Path(output).resolve()
    if (output/"manifest.json").exists():
        raise ValueError("演示目录已存在，请使用新目录")
    output.mkdir(parents=True, exist_ok=True)
    cfg = config_from()
    cfg.update({"sample_fps": 8, "window_seconds": 5, "history_windows": 4, "reference_body_length_px": 40, "expected_fish_count": 3})
    write_json(output/"config.json", cfg)
    recordings = []
    for block, split in enumerate(["train", "train", "val", "test"]):
        for number, scenario in enumerate(SCENARIOS):
            name = f"{split}_{block}_{number}"
            folder = output/"recordings"/name
            video = synth_video(folder, scenario, seed=100+block*10+number)
            result = analyze(video, folder/"analysis", cfg, detector="csv", detections=folder/"detections.csv", domain="synthetic", progress=False)
            recordings.append({"analysis": str((folder/"analysis"/"analysis.json").relative_to(output)), "labels": str((folder/"labels.csv").relative_to(output)), "group_id": name, "split": split})
            print(f"合成流程 {len(recordings)}/24：{name}，{len(result['windows'])}个时间窗", flush=True)
    manifest = output/"manifest.json"
    write_json(manifest, {"recordings": recordings})
    train(manifest, output/"model", cfg, backend=backend, epochs=epochs)
    # One held-out clip is shown; model evaluation uses all independent test clips.
    from .io import read_json
    report_folder = output/"recordings"/"test_3_3"/"analysis"
    analysis = read_json(report_folder/"analysis.json")
    predictions = infer(analysis, output/"model"/"model.joblib")
    write_json(report_folder/"predictions.json", predictions)
    # Copy just the preview to keep the top-level report portable and convenient.
    import shutil
    shutil.copyfile(report_folder/"preview.jpg", output/"preview.jpg")
    report = render_report(analysis, predictions, output/"report.html")
    print(f"报告：{report}", flush=True)
    return report
