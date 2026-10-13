import copy
import json
from pathlib import Path
import numpy as np
import pytest
from fishwatch.io import config_from, write_json
from fishwatch.features import FEATURES, track_features, summarize_window, causal_sequences, tree_vectors
from fishwatch.tracking import Detection, Tracker
from fishwatch.model import labels_from, infer, load_dataset
from fishwatch.report import render_report
from fishwatch.video import analyze
from fishwatch.demo import synth_video


def row(t, x, tid=1):
    return {"time_s": t, "track_id": tid, "cx": x, "cy": 50, "x1": x-5, "x2": x+5, "y1": 40, "y2": 60, "confidence": 1}


def test_constant_speed_and_gap_not_counted():
    cfg = config_from()
    cfg["reference_body_length_px"] = 10
    r = track_features([row(0, 10), row(0.1, 11), row(0.2, 12)], cfg, 3, 100, 100)
    assert r["speed_mean"] == pytest.approx(1)
    assert r["path_length"] == pytest.approx(0.2)
    assert r["displacement_ratio"] == pytest.approx(1)
    r = track_features([row(0, 10), row(5, 90)], cfg, 51, 100, 100)
    assert r["speed_mean"] is None
    assert r["path_length"] is None


def test_missing_fish_is_not_static_fish():
    cfg = config_from()
    w = summarize_window([{"fish": [], "time_s": t} for t in np.arange(0, 1, 0.1)], 0, 1, cfg, 100, 100, 0.1)
    assert not w["quality_ok"]
    assert w["features"]["speed_mean"] is None
    assert w["features"]["detection_coverage"] == 0


def test_expected_count_quality_gate():
    cfg = config_from()
    cfg["expected_fish_count"] = 3
    frames = [{"time_s": t, "fish": [row(t, 10+t)]} for t in np.arange(0, 1, 0.1)]
    w = summarize_window(frames, 0, 1, cfg, 100, 100, 0.1)
    assert not w["quality_ok"]
    assert w["features"]["detection_coverage"] == pytest.approx(1/3)


def test_temporal_features_never_use_future():
    w = [{"start_s": i, "end_s": i+1, "quality_ok": True, "features": {name: i for name in FEATURES}} for i in range(4)]
    a = causal_sequences(w, 3)
    w[3]["features"] = {name: 9999 for name in FEATURES}
    b = causal_sequences(w, 3)
    np.testing.assert_equal(a[:3], b[:3])
    assert np.isnan(a[0, 0]).all()
    assert tree_vectors(a).shape == (4, len(FEATURES)*5)


def test_gap_stops_history():
    w = [{"start_s": 0, "end_s": 1, "quality_ok": True, "features": {name: 1 for name in FEATURES}}, {"start_s": 5, "end_s": 6, "quality_ok": True, "features": {name: 2 for name in FEATURES}}]
    assert np.isnan(causal_sequences(w, 3)[1, :-1]).all()


def test_tracker_expiry_and_association():
    tracker = Tracker(config_from())
    first = tracker.update([Detection(np.array([0, 0, 10, 10]))], 0)
    second = tracker.update([Detection(np.array([1, 0, 11, 10]))], 0.1)
    third = tracker.update([Detection(np.array([1, 0, 11, 10]))], 2)
    assert first[0][0] == second[0][0]
    assert third[0][0] != first[0][0]


def test_overlapping_labels_rejected(tmp_path):
    path = tmp_path/"labels.csv"
    path.write_text("start_s,end_s,task,label,source\n0,10,oxygen,low,sensor\n5,12,oxygen,reference,sensor\n")
    with pytest.raises(ValueError, match="重叠"):
        labels_from(path)


def test_cross_split_group_rejected(tmp_path):
    cfg = config_from()
    analysis = {"schema_version": 1, "detector": "csv", "domain": "real", "recording_id": "a", "config": cfg, "motion_unit": "projected_bbox_widths/s", "windows": []}
    write_json(tmp_path/"a.json", analysis)
    (tmp_path/"l.csv").write_text("start_s,end_s,task,label,source\n")
    write_json(tmp_path/"manifest.json", {"recordings": [{"analysis": "a.json", "labels": "l.csv", "group_id": "tank1", "split": "train"}, {"analysis": "a.json", "labels": "l.csv", "group_id": "tank1", "split": "test"}]})
    with pytest.raises(ValueError, match="泄漏"):
        load_dataset(tmp_path/"manifest.json", cfg)


def test_domain_guard(tmp_path):
    import joblib
    model = {"schema_version": 1, "features": FEATURES, "domain": "synthetic"}
    joblib.dump(model, tmp_path/"model.joblib")
    with pytest.raises(ValueError, match="跨域"):
        infer({"domain": "real"}, tmp_path/"model.joblib")


def test_report_escapes_script_payload(tmp_path):
    analysis = {"domain": "synthetic", "video": "</script><script>alert(1)</script>", "windows": []}
    path = render_report(analysis, {"notice": "</script>"}, tmp_path/"report.html")
    text = path.read_text()
    assert "<script>alert(1)" not in text
    assert "\\u003c/script\\u003e" in text


def test_real_video_path_without_state_model(tmp_path):
    path = synth_video(tmp_path/"source", ("reference", "reference", "medium"), seed=1, seconds=4)
    cfg = config_from()
    cfg.update({"sample_fps": 8, "window_seconds": 2})
    result = analyze(path, tmp_path/"analysis", cfg, detections=path.parent/"detections.csv", overlay=True, progress=False)
    assert result["domain"] == "real"  # Public analyze path never silently claims synthetic labels.
    assert len(result["windows"]) == 2
    assert result["sampled_frames"] == 32
    assert result["windows"][0]["quality_ok"]
    assert infer(result)["status"] == "observations_only"
    assert (tmp_path/"analysis"/"tracks.mp4").stat().st_size > 100
    assert len(result["video_sha256"]) == 64


def test_config_invalid_roi():
    from fishwatch.cli import main
    assert main(["extract-frames", "missing.mp4", "--out", "unused", "--every-seconds", "0"]) == 2


def test_tcn_causal_and_missing_mask():
    pytest.importorskip("torch")
    import torch
    from fishwatch.temporal import make_network, fit_preprocess, transform
    x = np.ones((3, 4, len(FEATURES)), dtype=np.float32)
    x[0, 0] = np.nan
    prep = fit_preprocess(x)
    encoded = transform(x, prep)
    assert np.isfinite(encoded).all()
    assert np.all(encoded[0, 0, len(FEATURES):] == 0)
    net = make_network(encoded.shape[-1], {"oxygen": ["low", "reference"]})
    net.eval()
    tensor = torch.from_numpy(encoded)
    before = net.trunk(tensor.transpose(1, 2)).detach()
    tensor[:, -1] += 100
    after = net.trunk(tensor.transpose(1, 2)).detach()
    torch.testing.assert_close(before[:, :, :-1], after[:, :, :-1])


def test_tree_training_missing_tasks_and_quality_refusal(tmp_path):
    from fishwatch.model import train
    cfg = config_from()
    cfg["history_windows"] = 3
    records = []
    for split, count in [("train", 30), ("val", 15), ("test", 15)]:
        windows = []
        labels = ["start_s,end_s,task,label,source"]
        per_class = count//3
        for i in range(count):
            value = i//per_class
            windows.append({"start_s": i*10, "end_s": (i+1)*10, "quality_ok": True, "quality_reasons": [], "features": {name: value+(i%3)*0.01 for name in FEATURES}, "fish": []})
        for j, name in enumerate(["low", "reference", "high"]):
            labels.append(f"{j*per_class*10},{(j+1)*per_class*10},temperature,{name},synthetic_test")
        labels.append(f"0,{count*5},oxygen_high,reference,synthetic_test")
        labels.append(f"{count*5},{count*10},oxygen_high,high,synthetic_test")
        labels.append(f"0,{count*10},photoperiod,long,synthetic_test")
        analysis = {"schema_version": 1, "domain": "synthetic", "detector": "csv", "recording_id": split, "config": cfg, "motion_unit": "projected_bbox_widths/s", "windows": windows}
        write_json(tmp_path/f"{split}.json", analysis)
        (tmp_path/f"{split}.csv").write_text("\n".join(labels)+"\n")
        records.append({"analysis": f"{split}.json", "labels": f"{split}.csv", "group_id": split, "split": split})
    write_json(tmp_path/"manifest.json", {"recordings": records})
    model = train(tmp_path/"manifest.json", tmp_path/"model", cfg)
    assert set(model["tasks"]) == {"temperature"}
    assert "oxygen" in model["skipped"]
    assert "未启用" in model["skipped"]["oxygen_high"]
    assert model["evaluation"]["splits"]["test"]["groups"] == 1
    analysis["windows"][0]["quality_ok"] = False
    analysis["windows"][0]["quality_reasons"] = ["不可见"]
    pred = infer(analysis, tmp_path/"model"/"model.joblib")
    assert pred["windows"][0]["tasks"]["temperature"]["label"] == "unknown"
    assert pred["windows"][0]["tasks"]["temperature"]["probabilities"] is None
    exploratory = train(tmp_path/"manifest.json", tmp_path/"exploratory", cfg, include_exploratory=True)
    assert "oxygen_high" in exploratory["tasks"]
    assert "photoperiod" in exploratory["skipped"]  # No 24-hour context in these short clips.
    analysis["config"]["window_seconds"] = 3
    with pytest.raises(ValueError, match="不一致"):
        infer(analysis, tmp_path/"model"/"model.joblib")


def test_tcn_expands_receptive_field_for_day_context():
    pytest.importorskip("torch")
    from fishwatch.temporal import make_network
    net = make_network(40, {"photoperiod": ["short", "reference", "long"]}, history=144)
    receptive_field = 1+sum(2*block.padding for block in net.trunk)
    assert receptive_field >= 144
