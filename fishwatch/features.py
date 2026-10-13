"""Bounded, non-overlapping behavioral windows; missing observations stay missing."""
from collections import defaultdict
import numpy as np


FEATURES = [
    "speed_mean", "speed_p90", "speed_std", "acceleration_mean", "turn_rate",
    "path_length", "displacement_ratio", "surface_fraction", "bottom_fraction",
    "feeding_zone_fraction", "vertical_mean", "vertical_std", "nearest_distance",
    "dispersion", "count_mean", "count_std", "detection_coverage", "motion_pair_fraction",
    "direction_consistency", "net_turns", "activity_fraction", "wall_fraction",
    "shelter_fraction", "speed_baseline_z", "feeding_baseline_z"
]


def avg(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def inside(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def track_features(observations, cfg, expected_samples, width, height):
    a = sorted(observations, key=lambda r: r["time_s"])
    t = np.array([r["time_s"] for r in a], dtype=float)
    p = np.array([[r["cx"], r["cy"]] for r in a], dtype=float)
    scale = cfg["reference_body_length_px"] or float(np.median([r["x2"]-r["x1"] for r in a]))
    scale = max(scale, 1)
    roi = cfg["tank_roi"]
    x = (p[:, 0]/width-roi[0])/(roi[2]-roi[0])
    y = (p[:, 1]/height-roi[1])/(roi[3]-roi[1])
    dt = np.diff(t)
    valid = (dt > 0) & (dt <= cfg["max_gap_seconds"])
    delta = np.diff(p, axis=0) / scale
    lengths = np.linalg.norm(delta, axis=1)
    speeds = np.divide(lengths, dt, out=np.full_like(dt, np.nan), where=valid)
    usable = np.isfinite(speeds)
    turn, signed_angles, acceleration = [], [], []
    for j in range(1, len(delta)):
        if not (valid[j] and valid[j-1]):
            continue
        acceleration.append(abs(speeds[j]-speeds[j-1])/((dt[j]+dt[j-1])/2))
        if lengths[j] < 0.01 or lengths[j-1] < 0.01:
            continue
        u, v = delta[j-1], delta[j]
        angle = np.arctan2(u[0]*v[1]-u[1]*v[0], np.dot(u, v))
        signed_angles.append(float(angle))
        turn.append(abs(float(angle))/((dt[j]+dt[j-1])/2))
    path = float(lengths[valid].sum()) if valid.any() else None
    # Only a continuous track has a meaningful end-to-end/path ratio.
    ratio = float(np.linalg.norm(p[-1]-p[0])/scale/path) if path and valid.all() else None
    feed = cfg["feeding_roi"]
    margin = cfg.get("wall_margin", 0.10)
    shelter = cfg.get("shelter_roi")
    return {
        "track_id": a[0]["track_id"], "samples": len(a),
        "visible_fraction": min(len(a)/max(expected_samples, 1), 1),
        "mean_confidence": avg([r["confidence"] for r in a]),
        "speed_mean": avg(speeds),
        "speed_p90": float(np.percentile(speeds[usable], 90)) if usable.any() else None,
        "speed_std": float(np.std(speeds[usable])) if usable.any() else None,
        "activity_fraction": float(np.average(speeds[usable] > cfg.get("activity_speed_threshold", 0.15), weights=dt[usable])) if usable.any() else None,
        "wall_fraction": float(np.mean((x <= margin) | (x >= 1-margin))),
        "shelter_fraction": float(np.mean([inside(q, shelter) for q in zip(x, y)])) if shelter else None,
        "acceleration_mean": avg(acceleration), "turn_rate": avg(turn),
        "direction_consistency": float(abs(sum(signed_angles))/sum(abs(v) for v in signed_angles)) if sum(abs(v) for v in signed_angles) > 1e-8 else None,
        "net_turns": float(sum(signed_angles)/(2*np.pi)) if signed_angles else None,
        "path_length": path, "displacement_ratio": ratio,
        "surface_fraction": float(np.mean(y <= cfg["surface_fraction"])),
        "bottom_fraction": float(np.mean(y >= 1-cfg["bottom_fraction"])),
        "feeding_zone_fraction": float(np.mean([inside(q, feed) for q in zip(x, y)])),
        "vertical_mean": float(np.mean(y)), "vertical_std": float(np.std(y)),
        "valid_pairs": int(valid.sum()), "observed_pairs": max(len(a)-1, 0),
        "operculum_rate": None, "feeding_count": None,
        "note": "短轨迹编号；未实现呼吸/吞食动作检测，不以位置替代动作"
    }


def summarize_window(frames, start, end, cfg, width, height, sample_dt):
    tracks = defaultdict(list)
    counts, nearest, dispersion = [], [], []
    for frame in frames:
        rows = frame["fish"]
        counts.append(len(rows))
        for r in rows:
            tracks[r["track_id"]].append(r)
        if len(rows) >= 2:
            points = np.array([[r["cx"]/width, r["cy"]/height] for r in rows])
            distances = np.linalg.norm(points[:, None]-points[None], axis=-1)
            np.fill_diagonal(distances, np.inf)
            nearest.append(float(distances.min(axis=1).mean()))
            dispersion.append(float(np.linalg.norm(points-points.mean(axis=0), axis=1).mean()))
    expected = max(1, int(round((end-start)/sample_dt)))
    fish = [track_features(rows, cfg, expected, width, height) for rows in tracks.values()]
    visible = sum(c > 0 for c in counts)/expected
    if cfg["expected_fish_count"]:
        coverage = sum(min(c/cfg["expected_fish_count"], 1) for c in counts)/expected
    else:
        coverage = visible
    # Frame-level link ratio catches fragmentation even when one-frame tracklets dominate.
    valid_pairs = sum(f["valid_pairs"] for f in fish)
    all_observations = sum(counts)
    pair_fraction = valid_pairs/max(all_observations-max(counts, default=0), 1)
    stable = [f for f in fish if f["samples"] >= 3]
    features = {name: avg([f.get(name) for f in stable]) for name in FEATURES}
    features.update({
        "nearest_distance": avg(nearest), "dispersion": avg(dispersion),
        "count_mean": avg(counts), "count_std": float(np.std(counts)) if counts else None,
        "detection_coverage": min(coverage, 1), "motion_pair_fraction": min(pair_fraction, 1)
    })
    reasons = []
    if coverage < cfg["min_visible_fraction"]:
        reasons.append("可见覆盖不足")
    if pair_fraction < cfg["min_motion_pair_fraction"] or not stable:
        reasons.append("连续轨迹不足")
    if len(frames) < expected*0.8:
        reasons.append("采样覆盖不足")
    return {"start_s": float(start), "end_s": float(end), "samples": len(frames),
            "features": features, "quality_ok": not reasons, "quality_reasons": reasons,
            "fish": fish}


def causal_sequences(windows, history):
    """No future input, no implicit filling of missing frames with zero movement."""
    x = np.full((len(windows), history, len(FEATURES)), np.nan, dtype=np.float32)
    for i, window in enumerate(windows):
        for back in range(min(history, i+1)):
            prev = windows[i-back]
            # Require contiguous windows, even for imported analysis files.
            if back and abs(windows[i-back+1]["start_s"]-prev["end_s"]) > 1e-4:
                break
            if prev["quality_ok"]:
                x[i, history-1-back] = [np.nan if prev["features"].get(f) is None else prev["features"][f] for f in FEATURES]
    return x


def add_causal_baselines(windows, minimum=6, lookback=60):
    """Past valid windows only; short-term tank baseline, NOT a daily/individual norm.

    Values are descriptive deviations, never labels. Reset at discontinuities.
    Quality failures are excluded from both the baseline and the scored output.
    """
    past = []
    previous_end = None
    for window in windows:
        if previous_end is not None and abs(window["start_s"]-previous_end) > 1e-4:
            past = []
        previous_end = window["end_s"]
        for source, target in [("speed_mean", "speed_baseline_z"), ("feeding_zone_fraction", "feeding_baseline_z")]:
            value = window["features"].get(source)
            values = [p[source] for p in past[-lookback:] if p.get(source) is not None and np.isfinite(p[source])]
            score = None
            if window["quality_ok"] and value is not None and np.isfinite(value) and len(values) >= minimum:
                q25, median, q75 = np.quantile(values, [.25, .5, .75])
                # Constant/noisy baselines are not reliable reference scales.
                if q75-q25 > 1e-6:
                    score = float((value-median)/(q75-q25))
            window["features"][target] = score
        if window["quality_ok"]:
            past.append(dict(window["features"]))
            past = past[-lookback:]


def tree_vectors(sequences):
    """Current observation + past mean/std/slope; missingness is explicit."""
    out = []
    for seq in sequences:
        means, stds, slopes = [], [], []
        for col in seq.T:
            good = np.isfinite(col)
            means.append(float(np.mean(col[good])) if good.any() else np.nan)
            stds.append(float(np.std(col[good])) if good.any() else np.nan)
            slopes.append(float(np.polyfit(np.flatnonzero(good), col[good], 1)[0]) if good.sum() >= 2 else np.nan)
        out.append(np.r_[seq[-1], means, stds, slopes, np.mean(np.isfinite(seq), axis=0)])
    return np.asarray(out, dtype=np.float32)
