"""Latent state -> stochastic actions -> video/boxes -> observed features.

All coefficients are engineering hypotheses. No physiological calibration.
Truth/outcomes are stored separately and never added to model inputs.
"""
import itertools
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

from .io import config_from, read_json, write_json, write_csv
from .video import analyze
from .model import train, infer
from .report import render_report


STATES = ("hunger", "stress", "exploration", "fatigue")
DEFAULT_PARAMETERS = {
    "seconds": 48, "fps": 8, "window_seconds": 4,
    "simulated_minutes_per_video_second": 1.0,
    "hunger_rate_per_minute": .002, "satiety_per_intake": .12,
    "stress_recovery_per_minute": .004, "disturbance_gain": .18,
    "fatigue_gain_per_minute": .001, "rest_recovery_per_minute": .002,
    "action_interval_seconds": .75, "feeding_fraction": .625,
    "disturbance_fraction": .25, "dropout_probability": .04,
    "test_dropout_probability": .13, "occlusion_seconds": 1.0,
    "coefficient_jitter": .18, "test_coefficient_jitter": .38,
    "label_threshold": .5,
}


def parameters(path=None):
    p = dict(DEFAULT_PARAMETERS)
    if path:
        supplied = read_json(path)
        if set(supplied)-set(p):
            raise ValueError(f"未知模拟参数: {sorted(set(supplied)-set(p))}")
        p.update(supplied)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p.values()):
        raise ValueError("模拟参数必须是有限数值")
    for k in ("seconds", "fps", "window_seconds", "simulated_minutes_per_video_second", "action_interval_seconds"):
        if p[k] <= 0:
            raise ValueError(f"{k} 必须大于0")
    if p["seconds"] < 12 or p["fps"] < 3 or p["window_seconds"]*p["fps"] < 3:
        raise ValueError("至少12秒、3fps，且每窗口至少3帧")
    if p["seconds"]/p["window_seconds"] < 6:
        raise ValueError("至少需要6个行为窗口")
    if not np.isclose(p["seconds"]*p["fps"], round(p["seconds"]*p["fps"])):
        raise ValueError("seconds*fps 必须是整数帧数")
    for k in ("feeding_fraction", "disturbance_fraction", "label_threshold"):
        if not 0 < p[k] < 1:
            raise ValueError(f"{k} 必须在0和1之间")
    for k in ("dropout_probability", "test_dropout_probability", "coefficient_jitter", "test_coefficient_jitter"):
        if not 0 <= p[k] < 1:
            raise ValueError(f"{k} 必须在0和1之间（不含1）")
    for k, v in p.items():
        if v < 0:
            raise ValueError(f"{k} 不可为负数")
    if p["occlusion_seconds"] >= p["seconds"]:
        raise ValueError("遮挡时长须小于视频时长")
    return p


def update_state(state, dt_minutes, intake, disturbance, resting, p, rates=None):
    """Intake is consumed food, not offered food. All states remain bounded."""
    rates = rates or {"hunger": 1., "stress": 1., "fatigue": 1.}
    h, s, e, f = state
    return np.clip([
        h+p["hunger_rate_per_minute"]*rates["hunger"]*dt_minutes-p["satiety_per_intake"]*intake,
        s+p["disturbance_gain"]*disturbance-p["stress_recovery_per_minute"]*rates["stress"]*s*dt_minutes,
        e,
        f+(p["fatigue_gain_per_minute"]*(not resting)-p["rest_recovery_per_minute"]*resting)*rates["fatigue"]*dt_minutes,
    ], 0., 1.)


def action_probabilities(state, boldness, coefficients):
    h, s, e, f = state
    # Both escape and freezing can occur at high stress; activity is ambiguous.
    logits = np.array([1+3*e+boldness-2*f-s, 4*h-2*s-f,
                       3*f+1.2*s-1, 3*s-e, 2.5*s+boldness-f-1])
    logits = logits*coefficients
    weights = np.exp(logits-logits.max())
    return weights/weights.sum()  # explore, forage, rest, hide, dart


def generate_recording(folder, initial, seed, p, cfg, shifted=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    # Independent observation noise stream: dropout cannot change latent dynamics.
    camera = np.random.default_rng(seed+10_000_000)
    width, height = (600, 330) if shifted else (480, 270)
    dt = 1/p["fps"]
    jitter = p["test_coefficient_jitter"] if shifted else p["coefficient_jitter"]
    coefficients = rng.uniform(1-jitter, 1+jitter, 5)
    rates = dict(zip(("hunger", "stress", "fatigue"), rng.uniform(1-jitter, 1+jitter, 3)))
    boldness = float(rng.uniform(.15, .85))
    state = np.clip(np.asarray(initial)+rng.uniform(-.04, .04, 4), 0, 1)
    initial_state = state.copy()
    position = rng.uniform([.2, .25], [.7, .75])
    heading = rng.uniform(-np.pi, np.pi)
    speed_factor = rng.uniform(.75, 1.25)
    action, next_action, next_intake = 0, 0., 0.
    feed_at = p["seconds"]*p["feeding_fraction"]
    disturb_at = p["seconds"]*p["disturbance_fraction"]
    occlusion_at = p["seconds"]*.45
    food_target = np.array([.82, .18])
    shelter_target = np.array([.16, .75])
    video = folder/"synthetic.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), p["fps"], (width, height))
    if not writer.isOpened():
        raise ValueError("无法创建模拟视频")
    truth, detections, events = [], [], []
    events.extend([{"time_s": feed_at, "type": "food_offered"}, {"time_s": disturb_at, "type": "disturbance"}])
    try:
        for frame in range(round(p["seconds"]*p["fps"])):
            t = frame*dt
            if t >= next_action:
                action = int(rng.choice(5, p=action_probabilities(state, boldness, coefficients)))
                next_action = t+p["action_interval_seconds"]
                heading += rng.normal(0, 1.4)
            h, s, e, f = state
            if action in (1, 3):
                target = food_target if action == 1 else shelter_target
                vector = (target-position)*[width, height]
                heading = np.arctan2(vector[1], vector[0])+rng.normal(0, .18)
                distance = np.linalg.norm(vector)
            else:
                heading += rng.normal(0, .09 if action == 0 else .22)
                distance = math.inf
            speed = [1.3, 1.8, 0., 1.4, 4.5][action]*speed_factor*(1-.6*f)*40
            speed = min(speed, distance/dt)
            position += speed*dt*np.array([np.cos(heading)/width, np.sin(heading)/height])
            for axis in range(2):
                if not .06 <= position[axis] <= .94:
                    heading = np.pi-heading if axis == 0 else -heading
            position = np.clip(position, .06, .94)
            intake = 0
            if t >= feed_at and t >= next_intake and np.linalg.norm((position-food_target)*[width, height]) < 42:
                if rng.random() < h*(1-.8*s)*(1-.3*f):
                    intake = 1
                    next_intake = t+.8
                    events.append({"time_s": t, "type": "food_consumed", "amount": 1})
            disturbance = float(t <= disturb_at < t+dt)
            truth.append({"time_s": t, **dict(zip(STATES, map(float, state))), "boldness": boldness,
                          "behavior": ["explore", "forage", "rest", "hide", "dart"][action],
                          "cx": float(position[0]*width), "cy": float(position[1]*height)})
            state = update_state(state, dt*p["simulated_minutes_per_video_second"], intake, disturbance, action == 2, p, rates)
            image = np.full((height, width, 3), (55, 65, 30) if not shifted else (68, 48, 32), np.uint8)
            cv2.rectangle(image, (int(.05*width), int(.55*height)), (int(.3*width), int(.93*height)), (45, 90, 55), 2)
            if t >= feed_at:
                cv2.circle(image, tuple((food_target*[width, height]).astype(int)), 4, (70, 220, 230), -1)
            x, y = position*[width, height]
            cv2.ellipse(image, (round(x), round(y)), (20, 25), 0, 0, 360, (180, 210, 210), -1)
            cv2.circle(image, (round(x+10), round(y-5)), 2, (10, 30, 30), -1)
            cv2.putText(image, "SIMULATION - NOT BIOLOGICAL EVIDENCE", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .38, (220, 235, 220), 1)
            # Generate externally supplied boxes, not detector performance claims.
            dropped = camera.random() < p["test_dropout_probability" if shifted else "dropout_probability"]
            occluded = occlusion_at <= t < occlusion_at+p["occlusion_seconds"]
            if not dropped and not occluded:
                nx, ny = camera.normal(0, .15, 2)
                detections.append({"frame": frame, "x1": max(0, x+nx-20), "y1": max(0, y+ny-25),
                                   "x2": min(width, x+nx+20), "y2": min(height, y+ny+25), "confidence": .98})
            if occluded:
                cv2.rectangle(image, (round(x-23), round(y-28)), (round(x+23), round(y+28)), (30, 50, 35), -1)
            writer.write(image)
    finally:
        writer.release()
    write_csv(folder/"detections.csv", detections, ["frame", "x1", "y1", "x2", "y2", "confidence"])
    write_csv(folder/"latent_truth.csv", truth)
    write_json(folder/"events.json", events)
    write_json(folder/"generator.json", {"seed": seed, "parameters": p, "initial_state": dict(zip(STATES, initial_state)),
               "rates": rates, "action_coefficients": coefficients.tolist(), "boldness": boldness,
               "speed_factor": speed_factor, "shifted_test": shifted,
               "notice": "所有参数和隐藏状态均为人为假设；输入模型的只有视频框提取的行为特征。"})
    analysis = analyze(video, folder/"analysis", cfg, detections=folder/"detections.csv", domain="synthetic", progress=False)
    labels, window_truth = [], []
    times = np.array([r["time_s"] for r in truth])
    for window in analysis["windows"]:
        selected = [r for r, use in zip(truth, (times >= window["start_s"]-1e-6) & (times < window["end_s"]-1e-6)) if use]
        values = {key: float(np.mean([r[key] for r in selected])) for key in STATES}
        window_truth.append({"start_s": window["start_s"], "end_s": window["end_s"], "states": values})
        for key, value in values.items():
            labels.append({"start_s": window["start_s"], "end_s": window["end_s"], "task": f"sim_{key}",
                           "label": "high" if value >= p["label_threshold"] else "low", "source": "synthetic_latent_state_window_mean_not_biological_truth"})
    write_csv(folder/"labels.csv", labels)
    consumed = [r for r in events if r["type"] == "food_consumed"]
    summary = {"windows": window_truth, "events": events, "initial_state": dict(zip(STATES, initial_state)),
               "feeding_outcome": {"offered_at_s": feed_at, "first_intake_latency_s": consumed[0]["time_s"]-feed_at if consumed else None,
                                   "consumed_units": len(consumed), "censored": not bool(consumed)},
               "label_threshold": p["label_threshold"], "shifted_test": shifted}
    write_json(folder/"truth_summary.json", summary)
    return analysis


def simulate(output, backend="all", seed=42, parameter_path=None):
    p = parameters(parameter_path)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("模拟输出目录非空，请使用新目录")
    output.mkdir(parents=True, exist_ok=True)
    cfg = config_from()
    cfg.update(sample_fps=p["fps"], window_seconds=p["window_seconds"], history_windows=4,
               reference_body_length_px=40, expected_fish_count=1, shelter_roi=[.05, .55, .3, .93])
    write_json(output/"config.json", cfg)
    write_json(output/"simulation_parameters.json", p)
    records = []
    scenarios = list(itertools.product((.12, .85), repeat=4))
    # Full factorial in each split: hunger is not synonymous with stress/fatigue.
    for block, split in enumerate(("train", "val", "test")):
        for number, initial in enumerate(scenarios):
            name = f"{split}_{number:02d}"
            folder = output/"recordings"/name
            generate_recording(folder, initial, seed+block*10000+number, p, cfg, shifted=split == "test")
            records.append({"analysis": f"recordings/{name}/analysis/analysis.json", "labels": f"recordings/{name}/labels.csv",
                            "group_id": name, "split": split})
        print(f"模拟数据：{split} 16个独立会话已完成", flush=True)
    write_json(output/"manifest.json", {"recordings": records})
    results = {}
    for method in (["logistic", "gam", "hmm"] if backend == "all" else [backend]):
        model_dir = output/method/"model"
        model = train(output/"manifest.json", model_dir, cfg, backend=method, seed=seed)
        results[method] = model["evaluation"]
        for number in range(16):
            folder = output/"recordings"/f"test_{number:02d}"
            analysis = read_json(folder/"analysis"/"analysis.json")
            predictions = infer(analysis, model_dir/"model.joblib")
            target = output/method/f"test_{number:02d}"
            target.mkdir(parents=True, exist_ok=True)
            write_json(target/"predictions.json", predictions)
            shutil.copyfile(folder/"analysis"/"preview.jpg", target/"preview.jpg")
            render_report(analysis, predictions, target/"report.html", simulation=read_json(folder/"truth_summary.json"))
        print(f"{method}：训练、校准、偏移测试与16份对照报告完成", flush=True)
    write_json(output/"comparison.json", results)
    rows = []
    for method, evaluation in results.items():
        for task, score in evaluation["tasks"].items():
            rows.append(f"<tr><td>{method}</td><td>{task}</td><td>{score['accuracy']:.1%}</td><td>{score['brier_multiclass']:.3f}</td><td>{score['accepted_fraction']:.1%}</td></tr>")
    links = []
    for number, initial in enumerate(scenarios):
        name = " / ".join(f"{title}{'高' if value > .5 else '低'}" for title, value in zip(("饥饿", "紧张", "探索", "疲劳"), initial))
        links.append(f"<li>{name}："+" · ".join(f'<a href="{m}/test_{number:02d}/report.html">{m.upper()}</a>' for m in results)+"</li>")
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FishMood · 隐藏状态模拟研究</title><style>body{max-width:1050px;margin:40px auto;padding:0 24px;font:16px/1.8 system-ui;background:#f1f5f4;color:#183d43}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}a{color:#147d82}.notice{padding:16px;background:#fff1ce;border-radius:12px}li{margin:10px 0}</style>
<h1>FishMood · 隐藏状态模拟研究</h1><p class="notice">以下状态、轨迹和指标来自模拟。它们检验算法能否恢复生成器的设定，不是识别真实鱼心情的准确率。</p>
<p>四个隐藏状态 → 随机行为 → 视频与外部检测框 → 追踪特征 → 独立模型。每组16个独立会话；测试改变行为系数、画面尺度、背景与漏检率。隐藏真值、事件和摄食结果不进入模型输入。</p>
<h2>偏移测试结果</h2><p>所有合格测试窗口的准确率、Brier分数（越低越好）及达到接受阈值的比例。窗口不是独立动物；这不是跨生成机制验证。未按这些测试结果调参。</p>
<table><tr><th>模型</th><th>任务</th><th>准确率</th><th>Brier</th><th>接受率</th></tr>'''+"".join(rows)+"</table><h2>16种初始状态组合</h2><p>进入报告可切换时间窗，对比连续隐藏真值与模型类别概率；初始状态会随时间、惊扰和实际模拟进食变化。</p><ol>"+"".join(links)+"</ol></html>"
    (output/"index.html").write_text(page, encoding="utf-8")
    return output/"index.html"
