import csv
import json
import math
from pathlib import Path


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


DEFAULT_CONFIG = {
    "sample_fps": 10.0,
    "window_seconds": 10.0,
    "tank_roi": [0.0, 0.0, 1.0, 1.0],
    "feeding_roi": [0.65, 0.0, 1.0, 0.35],
    "surface_fraction": 0.15,
    "bottom_fraction": 0.15,
    "activity_speed_threshold": 0.15,
    "wall_margin": 0.10,
    "shelter_roi": None,
    "min_visible_fraction": 0.6,
    "min_motion_pair_fraction": 0.5,
    "reference_body_length_px": None,
    "max_gap_seconds": 0.5,
    "match_gate_body_lengths": 2.0,
    "min_detection_confidence": 0.35,
    "min_detection_area": 40,
    "max_detection_area_fraction": 0.1,
    "expected_fish_count": None,
    "history_windows": 6,
    "decision_threshold": 0.7,
    "min_training_windows": 20,
    "min_validation_windows": 5,
    "min_test_windows": 5
}


def config_from(path=None):
    cfg = dict(DEFAULT_CONFIG)
    if path:
        supplied = read_json(path)
        unknown = set(supplied) - set(cfg)
        if unknown:
            raise ValueError(f"未知配置项: {sorted(unknown)}")
        cfg.update(supplied)
    for key in ("sample_fps", "window_seconds", "max_gap_seconds", "match_gate_body_lengths", "activity_speed_threshold"):
        if not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} 必须为有限正数")
    for key in ("min_visible_fraction", "min_motion_pair_fraction", "decision_threshold", "min_detection_confidence"):
        if not 0 <= cfg[key] <= 1:
            raise ValueError(f"{key} 必须在0和1之间")
    for key in ("tank_roi", "feeding_roi", "shelter_roi"):
        box = cfg[key]
        if key == "shelter_roi" and box is None:
            continue
        if len(box) != 4 or not all(0 <= n <= 1 for n in box) or box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError(f"{key} 需要归一化的[x1,y1,x2,y2]")
    if cfg["history_windows"] < 1 or int(cfg["history_windows"]) != cfg["history_windows"]:
        raise ValueError("history_windows 必须为正整数")
    if cfg["reference_body_length_px"] is not None and cfg["reference_body_length_px"] <= 0:
        raise ValueError("参考体长必须大于0")
    if cfg["expected_fish_count"] is not None and (cfg["expected_fish_count"] < 1 or int(cfg["expected_fish_count"]) != cfg["expected_fish_count"]):
        raise ValueError("expected_fish_count 必须为正整数或null")
    for key in ("surface_fraction", "bottom_fraction"):
        if not 0 < cfg[key] < 1:
            raise ValueError(f"{key} 必须在0和1之间")
    if cfg["surface_fraction"] + cfg["bottom_fraction"] >= 1:
        raise ValueError("表层和底层区域不可重叠")
    if not 0 < cfg["wall_margin"] < 0.5:
        raise ValueError("wall_margin 必须在0和0.5之间")
    return cfg
