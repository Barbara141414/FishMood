import csv
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, log_loss, average_precision_score
from .features import FEATURES, causal_sequences, tree_vectors
from .io import read_json, write_json


TASKS = {
    **{f"sim_{key}": {"title": f"{title}（模拟隐藏状态）", "classes": ["low", "high"],
                        "labels": ["低", "高"], "synthetic_only": True}
       for key, title in [("hunger", "饥饿驱动"), ("stress", "紧张程度"),
                          ("exploration", "探索倾向"), ("fatigue", "疲劳程度")]},
    "temperature": {"title": "温度暴露估计", "classes": ["low", "reference", "high"], "labels": ["低温暴露", "参考温度", "高温暴露"]},
    "oxygen": {"title": "溶氧暴露估计", "classes": ["low", "reference"], "labels": ["低氧暴露", "参考溶氧"]},
    "feeding_response": {"title": "摄食响应估计（非饥饿诊断）", "classes": ["low", "medium", "high"], "labels": ["弱响应", "中等响应", "强响应"]},
    "oxygen_high": {"title": "高氧暴露估计（探索性）", "classes": ["reference", "high"], "labels": ["参考溶氧", "高氧暴露"], "exploratory": True},
    "photoperiod": {"title": "光周期暴露估计（探索性）", "classes": ["short", "reference", "long"], "labels": ["短光周期", "参考光周期", "长光周期"], "exploratory": True, "min_history_seconds": 86400}
}


def contract(analysis):
    cfg = analysis["config"]
    return {k: cfg[k] for k in ["window_seconds", "surface_fraction", "bottom_fraction", "feeding_roi"]} | {
        "motion_unit": analysis["motion_unit"], "activity_speed_threshold": cfg.get("activity_speed_threshold", .15),
        "wall_margin": cfg.get("wall_margin", .1), "shelter_roi": cfg.get("shelter_roi")}


def labels_from(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        needed = {"start_s", "end_s", "task", "label", "source"}
        if not needed <= set(reader.fieldnames or []):
            raise ValueError("标签CSV需含start_s,end_s,task,label,source")
        intervals = {task: [] for task in TASKS}
        for row in reader:
            task = row["task"]
            if task not in TASKS or row["label"] not in TASKS[task]["classes"]:
                raise ValueError(f"未知任务/标签: {task}/{row['label']}")
            start, end = float(row["start_s"]), float(row["end_s"])
            if not np.isfinite([start, end]).all() or not 0 <= start < end or not row["source"].strip():
                raise ValueError("标签需要合法时间区间与独立来源source")
            intervals[task].append((start, end, row["label"], row["source"]))
    for task, entries in intervals.items():
        entries.sort()
        if any(a[1] > b[0] for a, b in zip(entries, entries[1:])):
            raise ValueError(f"{task} 标签时间区间重叠")
    return intervals


def load_dataset(manifest, cfg):
    manifest = Path(manifest).resolve()
    data = read_json(manifest)
    records = data.get("recordings", [])
    if not records:
        raise ValueError("manifest需要非空recordings")
    buckets = {split: {"x": [], "y": {t: [] for t in TASKS}, "rows": [], "excluded": 0} for split in ["train", "val", "test"]}
    group_splits, video_splits, seen_analyses, domains, contracts = {}, {}, set(), set(), []
    provenance = []
    for record in records:
        split, group = record["split"], str(record["group_id"])
        if split not in buckets or not group:
            raise ValueError("split需为train/val/test，group_id不可为空")
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"数据泄漏：group_id={group}跨数据集")
        group_splits[group] = split
        analysis_path = (manifest.parent/record["analysis"]).resolve()
        if analysis_path in seen_analyses:
            raise ValueError("重复分析文件；请一个recording只列一次")
        seen_analyses.add(analysis_path)
        analysis = read_json(analysis_path)
        if analysis.get("schema_version") != 1:
            raise ValueError("分析schema版本不兼容")
        if analysis["detector"] == "motion":
            raise ValueError("运动前景检测不能作为状态模型训练数据；请使用人工框或训练好的鱼类检测器")
        vid = analysis.get("video_sha256", analysis["recording_id"])
        if vid in video_splits and video_splits[vid] != split:
            raise ValueError("数据泄漏：相同源视频跨数据集")
        video_splits[vid] = split
        domains.add(analysis["domain"])
        contracts.append(contract(analysis))
        if contracts[-1] != contracts[0]:
            raise ValueError("特征定义不一致：窗口、分区或速度单位不同")
        label_path = (manifest.parent/record["labels"]).resolve()
        intervals = labels_from(label_path)
        if analysis["domain"] != "synthetic" and any(intervals[t] for t, spec in TASKS.items() if spec.get("synthetic_only")):
            raise ValueError("模拟隐藏状态标签只能用于synthetic数据，不能作为真实鱼心理真值")
        windows = analysis["windows"]
        seqs = causal_sequences(windows, cfg["history_windows"])
        bucket = buckets[split]
        for window, seq in zip(windows, seqs):
            if not window["quality_ok"]:
                bucket["excluded"] += 1
                continue
            values = {}
            for task, entries in intervals.items():
                labels = [label for start, end, label, source in entries if start <= window["start_s"]+1e-6 and end >= window["end_s"]-1e-6]
                values[task] = TASKS[task]["classes"].index(labels[0]) if labels else -1
                if TASKS[task].get("min_history_seconds"):
                    required = TASKS[task]["min_history_seconds"]
                    sufficient = cfg["history_windows"]*analysis["config"]["window_seconds"] >= required and window["end_s"]-windows[0]["start_s"] >= required
                    # Full history is required for light-cycle labels; short clips cannot certify a day.
                    if not sufficient or not np.isfinite(seq[:, FEATURES.index("speed_mean")]).all():
                        values[task] = -1
            if all(v < 0 for v in values.values()):
                bucket["excluded"] += 1
                continue
            bucket["x"].append(seq)
            for task, value in values.items():
                bucket["y"][task].append(value)
            bucket["rows"].append({"group_id": group, "analysis": str(analysis_path), "start_s": window["start_s"], "end_s": window["end_s"]})
        provenance.append({"analysis": str(analysis_path), "labels": str(label_path), "group_id": group, "split": split, "domain": analysis["domain"]})
    if len(domains) != 1:
        raise ValueError("合成数据与真实数据不能混合训练")
    for split, bucket in buckets.items():
        minimum = cfg[{"train": "min_training_windows", "val": "min_validation_windows", "test": "min_test_windows"}[split]]
        if len(bucket["x"]) < minimum:
            raise ValueError(f"{split} 有效标注窗口仅{len(bucket['x'])}，至少需要{minimum}；检查标签全覆盖、轨迹质量与独立组")
        bucket["x"] = np.array(bucket["x"], dtype=np.float32)
        bucket["y"] = {k: np.array(v, dtype=int) for k, v in bucket["y"].items()}
    return buckets, domains.pop(), contracts[0], provenance


def calibrate(logits, y):
    if len(np.unique(y)) < 2:
        return 1.0
    result = minimize_scalar(lambda logt: log_loss(y, softmax(logits/np.exp(logt), axis=1), labels=np.arange(logits.shape[1])), bounds=(-2, 3), method="bounded")
    return float(np.exp(result.x))


def metrics(y, probabilities, classes, threshold):
    prediction = probabilities.argmax(axis=1)
    keep = probabilities.max(axis=1) >= threshold
    onehot = np.eye(len(classes))[y]
    aps = {}
    for j, name in enumerate(classes):
        target = y == j
        aps[name] = float(average_precision_score(target, probabilities[:, j])) if target.any() and not target.all() else None
    return {"n_windows": len(y), "accuracy": float(np.mean(prediction == y)),
            "classification": classification_report(y, prediction, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, prediction, labels=list(range(len(classes)))).tolist(),
            "classes": classes, "brier_multiclass": float(np.mean(np.sum((probabilities-onehot)**2, axis=1))),
            "log_loss": float(log_loss(y, probabilities, labels=list(range(len(classes))))),
            "average_precision": aps, "accepted_fraction": float(keep.mean()),
            "accepted_accuracy": float(np.mean(prediction[keep] == y[keep])) if keep.any() else None,
            "threshold": threshold}


def train(manifest, output, cfg, backend="logistic", epochs=60, seed=42, include_exploratory=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"model.joblib").exists():
        raise ValueError("模型已存在，请选择新的输出目录")
    data, domain, feature_contract, provenance = load_dataset(manifest, cfg)
    active, skipped = {}, {}
    for task, spec in TASKS.items():
        if spec.get("synthetic_only") and domain != "synthetic":
            skipped[task] = "仅适用于模拟隐藏状态，不支持真实鱼心理诊断"
            continue
        if spec.get("exploratory") and not include_exploratory:
            skipped[task] = "探索性任务未启用；需要独立标签和物种验证"
            continue
        present = set(data["train"]["y"][task][data["train"]["y"][task] >= 0])
        if present != set(range(len(spec["classes"]))):
            skipped[task] = "训练集未覆盖全部预定义类别"
            continue
        if set(data["val"]["y"][task][data["val"]["y"][task] >= 0]) != present:
            skipped[task] = "验证集未覆盖全部类别，不能可靠校准"
            continue
        active[task] = spec["classes"]
    if not active:
        raise ValueError("没有任务同时在训练、验证集中覆盖所有类别")
    model = {"schema_version": 1, "backend": backend, "domain": domain,
             "features": FEATURES, "history_windows": cfg["history_windows"],
             "contract": feature_contract, "tasks": active, "skipped": skipped,
             "calibration": {}, "threshold": cfg["decision_threshold"], "seed": seed}
    logits = {"val": {}, "test": {}}
    if backend == "tree":
        vectors = {s: tree_vectors(b["x"]) for s, b in data.items()}
        model["estimators"] = {}
        for task, classes in active.items():
            y = data["train"]["y"][task]
            mask = y >= 0
            clf = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, min_samples_leaf=4, l2_regularization=1.0, class_weight="balanced", early_stopping=False, random_state=seed)
            clf.fit(vectors["train"][mask], y[mask])
            model["estimators"][task] = clf
            for split in logits:
                logits[split][task] = np.log(np.clip(clf.predict_proba(vectors[split]), 1e-8, 1))
    elif backend in {"logistic", "gam", "hmm"}:
        from .classical import vectors, fit_classical, probabilities, explain
        vx = {s: vectors(b["x"]) for s, b in data.items()}
        model["estimators"] = {}
        explanations = {}
        for task in active:
            y = data["train"]["y"][task]
            estimator = fit_classical(backend, vx["train"], y, data["train"]["rows"])
            model["estimators"][task] = estimator
            explanations[task] = explain(estimator, backend, vx["train"][y >= 0])
            for split in logits:
                logits[split][task] = np.log(np.clip(probabilities(estimator, vx[split], data[split]["rows"]), 1e-300, 1))
        write_json(output/"explanations.json", explanations)
    elif backend == "tcn":
        from .temporal import fit_tcn, tcn_logits
        model["tcn"] = fit_tcn(data["train"]["x"], {k: data["train"]["y"][k] for k in active}, data["val"]["x"], {k: data["val"]["y"][k] for k in active}, active, epochs, seed)
        for split in logits:
            logits[split] = tcn_logits(model["tcn"], data[split]["x"], active)
    else:
        raise ValueError("backend必须为logistic/gam/hmm/tree/tcn")
    evaluation = {"domain": domain, "backend": backend, "skipped_tasks": skipped, "tasks": {},
                  "splits": {s: {"windows": len(b["x"]), "excluded_windows": b["excluded"], "groups": len(set(r["group_id"] for r in b["rows"]))} for s, b in data.items()},
                  "notice": "合成测试仅验证代码流程，不代表真实鱼识别准确率。" if domain == "synthetic" else "指标来自独立group测试集；是否跨缸取决于用户提供的group_id。",
                  "limitations": ["阈值为工程初值，未进行现场误报/日与检测延迟验收。", "温度缩放使用验证集；测试集不参与训练/校准。", "模型学习暴露标签与视频关联，不能证实病因或生理损伤；高氧/光周期研究头需额外验证。"]}
    for task, classes in active.items():
        val_y = data["val"]["y"][task]
        valid = val_y >= 0
        temperature = calibrate(logits["val"][task][valid], val_y[valid])
        model["calibration"][task] = temperature
        test_y = data["test"]["y"][task]
        mask = test_y >= 0
        if mask.any():
            probabilities = softmax(logits["test"][task][mask]/temperature, axis=1)
            evaluation["tasks"][task] = metrics(test_y[mask], probabilities, classes, model["threshold"])
            group_results = {}
            for group in sorted({r["group_id"] for r in data["test"]["rows"]}):
                gm = np.array([r["group_id"] == group for r in data["test"]["rows"]]) & mask
                if gm.any():
                    group_results[group] = metrics(test_y[gm], softmax(logits["test"][task][gm]/temperature, axis=1), classes, model["threshold"])
            evaluation["tasks"][task]["by_group"] = group_results
        else:
            evaluation["tasks"][task] = {"status": "没有独立测试标签，未评估"}
    model["evaluation"] = evaluation
    joblib.dump(model, output/"model.joblib")
    write_json(output/"evaluation.json", evaluation)
    write_json(output/"provenance.json", provenance)
    write_json(output/"model_card.json", {k: v for k, v in model.items() if k not in ["estimators", "tcn"]})
    if backend == "tcn":
        write_json(output/"training_log.json", model["tcn"]["training_log"])
    return model


def infer(analysis, model_path=None):
    visible_tasks = {t: spec for t, spec in TASKS.items() if analysis["domain"] == "synthetic" or not spec.get("synthetic_only")}
    if model_path is None:
        return {"domain": analysis["domain"], "status": "observations_only", "tasks": visible_tasks, "windows": [], "notice": "未加载训练模型：只展示行为观测，不推断饥饿或环境原因。"}
    # Only load locally trusted model files: joblib uses Python pickle.
    model = joblib.load(model_path)
    if model.get("schema_version") != 1 or model.get("features") != FEATURES:
        raise ValueError("模型schema/特征版本不兼容；本版扩充行为特征，请重新分析视频并训练模型")
    if model["domain"] != analysis["domain"]:
        raise ValueError("拒绝跨域使用：合成演示模型不能预测真实鱼视频，反之亦然")
    if analysis["domain"] != "synthetic" and any(TASKS[t].get("synthetic_only") for t in model["tasks"]):
        raise ValueError("模拟隐藏状态模型不能用于真实鱼")
    if analysis["detector"] == "motion":
        raise ValueError("前景运动模式不具备可靠鱼检测能力，仅允许行为预览")
    if model["contract"] != contract(analysis):
        raise ValueError("分析与训练的窗口、分区或速度单位不一致")
    seqs = causal_sequences(analysis["windows"], model["history_windows"])
    if model["backend"] == "tree":
        x = tree_vectors(seqs)
        logits = {k: np.log(np.clip(m.predict_proba(x), 1e-8, 1)) for k, m in model["estimators"].items()}
    elif model["backend"] in {"logistic", "gam", "hmm"}:
        from .classical import vectors, probabilities
        rows = [{"analysis": "current", "start_s": w["start_s"], "end_s": w["end_s"], "quality_ok": w["quality_ok"]} for w in analysis["windows"]]
        logits = {k: np.log(np.clip(probabilities(m, vectors(seqs), rows), 1e-300, 1)) for k, m in model["estimators"].items()}
    else:
        from .temporal import tcn_logits
        logits = tcn_logits(model["tcn"], seqs, model["tasks"])
    probabilities = {k: softmax(v/model["calibration"][k], axis=1) for k, v in logits.items()}
    results = []
    for i, window in enumerate(analysis["windows"]):
        tasks = {}
        for task, classes in model["tasks"].items():
            p = probabilities[task][i]
            best = int(p.argmax())
            evidence_ok = window["quality_ok"]
            accepted = evidence_ok and float(p[best]) >= model["threshold"]
            reason = "；".join(window["quality_reasons"]) if not window["quality_ok"] else "模型概率低于阈值"
            required = TASKS[task].get("min_history_seconds", 0)
            if required and (window["end_s"]-analysis["windows"][0]["start_s"] < required or not np.isfinite(seqs[i, :, FEATURES.index("speed_mean")]).all()):
                accepted, evidence_ok, reason = False, False, "没有完整且有效的至少24小时行为历史"
            tasks[task] = {"label": classes[best] if accepted else "unknown", "confidence": float(p[best]) if evidence_ok else None,
                           "probabilities": {k: float(v) for k, v in zip(classes, p)} if evidence_ok else None,
                           "reason": None if accepted else reason}
        if tasks.get("oxygen", {}).get("label") == "low" and tasks.get("oxygen_high", {}).get("label") == "high":
            for task in ["oxygen", "oxygen_high"]:
                tasks[task]["label"] = "unknown"
                tasks[task]["reason"] = "低氧与高氧研究头输出矛盾，需独立溶氧测量"
        results.append({"start_s": window["start_s"], "end_s": window["end_s"], "tasks": tasks})
    return {"domain": analysis["domain"], "status": "model_estimates", "backend": model["backend"], "tasks": visible_tasks,
            "trained_tasks": list(model["tasks"]), "skipped_tasks": model["skipped"], "windows": results, "evaluation": model["evaluation"],
            "notice": "合成数据训练的演示推断，不具有生物学意义。" if model["domain"] == "synthetic" else "缸级暴露/响应估计，非病因确诊；未训练的任务不输出预测。"}
