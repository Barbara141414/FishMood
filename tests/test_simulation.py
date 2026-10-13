import copy
import numpy as np
import pytest

from fishwatch.features import add_causal_baselines, track_features
from fishwatch.io import config_from, read_json, write_json
from fishwatch.model import load_dataset, infer
from fishwatch.simulation import parameters, update_state, generate_recording, simulate


def test_hunger_changes_with_consumption_not_offering_and_states_bounded():
    p = parameters()
    state = np.array([.8, .7, .6, .5])
    fasting = update_state(state, 1, 0, 0, False, p)
    eating = update_state(state, 1, 1, 0, False, p)
    assert fasting[0] > state[0] > eating[0]
    assert update_state(state, 1, 0, 1, False, p)[1] > fasting[1]
    extreme = update_state(state, 100000, 10000, 10000, True, p)
    assert np.all((extreme >= 0) & (extreme <= 1))


def test_baseline_is_causal_and_resets_at_gap():
    windows = [{"start_s": i, "end_s": i+1, "quality_ok": True,
                "features": {"speed_mean": float(i), "feeding_zone_fraction": i/20}} for i in range(12)]
    prefix = copy.deepcopy(windows[:9])
    add_causal_baselines(windows)
    add_causal_baselines(prefix)
    assert windows[:9] == prefix
    assert all(w["features"]["speed_baseline_z"] is None for w in windows[:6])
    windows[10]["start_s"] = 100
    add_causal_baselines(windows)
    assert windows[10]["features"]["speed_baseline_z"] is None


def test_activity_missing_is_not_rest_or_hiding():
    cfg = config_from()
    cfg.update(reference_body_length_px=10, shelter_roi=[0, 0, .3, 1])
    def row(t):
        return {"time_s": t, "track_id": 1, "cx": 10, "cy": 50,
                "x1": 5, "x2": 15, "confidence": 1}
    observed = track_features([row(0), row(.1), row(.2)], cfg, 3, 100, 100)
    assert observed["activity_fraction"] == 0
    assert observed["shelter_fraction"] == 1
    missing = track_features([row(0), row(10)], cfg, 100, 100, 100)
    assert missing["activity_fraction"] is None


def test_simulation_labels_rejected_in_real_domain(tmp_path):
    cfg = config_from()
    write_json(tmp_path/"a.json", {"schema_version": 1, "detector": "csv", "domain": "real",
               "recording_id": "a", "config": cfg, "motion_unit": "reference_body_lengths/s", "windows": []})
    (tmp_path/"labels.csv").write_text("start_s,end_s,task,label,source\n0,10,sim_hunger,high,simulated\n")
    write_json(tmp_path/"manifest.json", {"recordings": [{"analysis": "a.json", "labels": "labels.csv", "group_id": "a", "split": "train"}]})
    with pytest.raises(ValueError, match="心理真值"):
        load_dataset(tmp_path/"manifest.json", cfg)


def test_generator_truth_independent_of_camera_dropout(tmp_path):
    p = parameters()
    p.update(seconds=12, window_seconds=2, fps=4)
    cfg = config_from()
    cfg.update(sample_fps=4, window_seconds=2, reference_body_length_px=40, expected_fish_count=1)
    a = generate_recording(tmp_path/"a", [.85, .85, .12, .12], 6, p, cfg)
    p["dropout_probability"] = .7
    generate_recording(tmp_path/"b", [.85, .85, .12, .12], 6, p, cfg)
    assert (tmp_path/"a"/"latent_truth.csv").read_bytes() == (tmp_path/"b"/"latent_truth.csv").read_bytes()
    assert a["domain"] == "synthetic"
    assert not any("hunger" in key or "stress" in key for w in a["windows"] for key in w["features"])
    assert "simulation" not in a


def test_full_simulation_pipeline_and_truth_not_input(tmp_path):
    p = parameters()
    p.update(seconds=12, window_seconds=2, fps=4)
    write_json(tmp_path/"parameters.json", p)
    report = simulate(tmp_path/"run", backend="logistic", parameter_path=tmp_path/"parameters.json")
    assert report.is_file()
    root = report.parent
    evaluation = read_json(root/"logistic"/"model"/"evaluation.json")
    assert set(evaluation["tasks"]) == {"sim_hunger", "sim_stress", "sim_exploration", "sim_fatigue"}
    assert evaluation["splits"]["test"]["groups"] == 16
    analysis = read_json(root/"recordings"/"test_00"/"analysis"/"analysis.json")
    before = infer(analysis, root/"logistic"/"model"/"model.joblib")
    # Altering external truth/outcomes does not alter inference.
    write_json(root/"recordings"/"test_00"/"truth_summary.json", {"hunger": 999})
    after = infer(analysis, root/"logistic"/"model"/"model.joblib")
    assert before == after
    assert "模拟真值与推断对照" in (root/"logistic"/"test_00"/"report.html").read_text()
    with pytest.raises(ValueError, match="非空"):
        simulate(root, backend="logistic")


@pytest.mark.parametrize("key,value", [("fps", 0), ("seconds", float("nan")), ("dropout_probability", 1), ("label_threshold", 0)])
def test_invalid_parameters(tmp_path, key, value):
    import json
    path = tmp_path/"p.json"
    path.write_text(json.dumps({key: value}))
    with pytest.raises(ValueError):
        parameters(path)
