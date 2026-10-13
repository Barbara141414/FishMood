import argparse
import sys
from pathlib import Path
from .io import config_from, write_json, read_json


def parser():
    p = argparse.ArgumentParser(description="小鱼：神仙鱼长视频行为分析与可验证状态建模")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("init", help="生成配置、标签和训练清单模板")
    a.add_argument("--out", default="project_config")
    a = sub.add_parser("demo", help="合成视频→追踪→特征→训练→测试→报告，非真实识别验证")
    a.add_argument("--out", default="outputs/demo")
    a.add_argument("--backend", choices=["logistic", "gam", "hmm", "tree", "tcn"], default="logistic")
    a.add_argument("--epochs", type=int, default=40)
    a = sub.add_parser("simulate", help="隐藏状态→行为轨迹→三版模型→独立模拟测试（非真实心理诊断）")
    a.add_argument("--out", default="outputs/mood_simulation")
    a.add_argument("--backend", choices=["all", "logistic", "gam", "hmm"], default="all")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--parameters", help="模拟参数JSON；省略时使用记录在输出中的工程默认值")
    a = sub.add_parser("analyze", help="流式分析真实视频，默认只输出观测")
    a.add_argument("video")
    a.add_argument("--out", required=True)
    a.add_argument("--config")
    det = a.add_mutually_exclusive_group(required=True)
    det.add_argument("--detections", help="人工/外部检测CSV，原视频frame编号")
    det.add_argument("--weights", help="本地已训练的神仙鱼YOLO权重")
    det.add_argument("--motion-preview", action="store_true", help="实验性前景预览，不允许状态推理")
    a.add_argument("--class-id", type=int, default=0)
    a.add_argument("--overlay", action="store_true", help="另存采样帧率的轨迹标注视频")
    a.add_argument("--model", help="本地可信状态模型；必须与真实域及特征定义匹配")
    a = sub.add_parser("train", help="按独立group的train/val/test清单训练状态模型")
    a.add_argument("--manifest", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--config")
    a.add_argument("--backend", choices=["logistic", "gam", "hmm", "tree", "tcn"], default="logistic")
    a.add_argument("--epochs", type=int, default=60)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--include-exploratory", action="store_true", help="允许训练有独立标签的高氧/光周期研究头；光周期要求至少24小时有效历史")
    a = sub.add_parser("predict", help="对已有行为特征进行状态推理并生成报告")
    a.add_argument("analysis")
    a.add_argument("--model", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("report", help="重新生成离线HTML报告")
    a.add_argument("analysis")
    a.add_argument("--predictions")
    a.add_argument("--out", required=True)
    a = sub.add_parser("extract-frames", help="抽取检测器训练标注所需图片")
    a.add_argument("video")
    a.add_argument("--out", required=True)
    a.add_argument("--every-seconds", type=float, default=5)
    a = sub.add_parser("train-detector", help="用已标注YOLO数据训练鱼检测器（可选依赖）")
    a.add_argument("--data", required=True)
    a.add_argument("--initial-weights", required=True, help="本地YOLO权重或模型yaml")
    a.add_argument("--out", required=True)
    a.add_argument("--epochs", type=int, default=60)
    a.add_argument("--device", default="cpu")
    return p


def init_files(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = [out/"config.json", out/"labels.csv", out/"manifest.json", out/"detection_dataset.yaml", out/"long_history_config.json"]
    if any(p.exists() for p in files):
        raise ValueError("模板已存在，请使用新目录")
    write_json(files[0], config_from())
    files[1].write_text("start_s,end_s,task,label,source\n", encoding="utf-8")
    write_json(files[2], {"recordings": [
        {"analysis": "../outputs/tankA_session1/analysis.json", "labels": "labels_train.csv", "group_id": "tank_A", "split": "train"},
        {"analysis": "../outputs/tankB_session1/analysis.json", "labels": "labels_val.csv", "group_id": "tank_B", "split": "val"},
        {"analysis": "../outputs/tankC_session1/analysis.json", "labels": "labels_test.csv", "group_id": "tank_C", "split": "test"}
    ]})
    files[3].write_text("# Replace path with an absolute path to your annotated dataset.\npath: /absolute/path/to/fish_dataset\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: angelfish\n", encoding="utf-8")
    long_cfg = config_from()
    long_cfg.update(window_seconds=600, history_windows=144, sample_fps=5)
    write_json(files[4], long_cfg)
    print(f"模板：{out.resolve()}")


def extract_frames(video, output, every):
    import cv2
    import math
    if not math.isfinite(every) or every <= 0:
        raise ValueError("抽帧间隔必须为有限正数")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("无法打开视频")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("视频帧率无效")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("frame_*.jpg")):
        cap.release()
        raise ValueError("目录已有抽帧图片，请使用新目录")
    step, i, saved = max(1, round(every*fps)), 0, 0
    try:
        while cap.grab():
            if i % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError(f"第{i}帧解码失败")
                if not cv2.imwrite(str(output/f"frame_{i:09d}.jpg"), frame):
                    raise ValueError("保存抽帧失败")
                saved += 1
            i += 1
    finally:
        cap.release()
    print(f"已抽取 {saved} 张图片，文件名包含原始帧编号；请按完整视频/鱼缸划分数据，勿随机拆相邻帧。")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if hasattr(args, "epochs") and args.epochs < 1:
            raise ValueError("epochs必须大于0")
        if args.command == "init":
            init_files(args.out)
        elif args.command == "demo":
            from .demo import demo
            demo(args.out, args.backend, args.epochs)
        elif args.command == "simulate":
            from .simulation import simulate
            print(f"模拟对照报告：{simulate(args.out, args.backend, args.seed, args.parameters)}")
        elif args.command == "analyze":
            from .video import analyze
            from .model import infer
            from .report import render_report
            cfg = config_from(args.config)
            if args.motion_preview and args.model:
                raise ValueError("运动前景仅可预览，不能加载状态模型")
            result = analyze(args.video, args.out, cfg, detector="csv" if args.detections else "yolo" if args.weights else "motion", detections=args.detections, weights=args.weights, class_id=args.class_id, overlay=args.overlay)
            predictions = infer(result, args.model)
            write_json(Path(args.out)/"predictions.json", predictions)
            report = render_report(result, predictions, Path(args.out)/"report.html")
            print(f"完成：{report}")
        elif args.command == "train":
            from .model import train
            train(args.manifest, args.out, config_from(args.config), args.backend, args.epochs, args.seed, args.include_exploratory)
            print(f"模型与独立测试评估：{Path(args.out).resolve()}")
        elif args.command == "predict":
            import shutil
            from .model import infer
            from .report import render_report
            analysis = read_json(args.analysis)
            predictions = infer(analysis, args.model)
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            write_json(out/"predictions.json", predictions)
            preview = Path(args.analysis).parent/"preview.jpg"
            if preview.exists() and preview.resolve() != (out/"preview.jpg").resolve():
                shutil.copyfile(preview, out/"preview.jpg")
            print(render_report(analysis, predictions, out/"report.html"))
        elif args.command == "report":
            from .model import infer
            from .report import render_report
            analysis = read_json(args.analysis)
            predictions = read_json(args.predictions) if args.predictions else infer(analysis)
            print(render_report(analysis, predictions, args.out))
        elif args.command == "extract-frames":
            extract_frames(args.video, args.out, args.every_seconds)
        elif args.command == "train-detector":
            if not Path(args.initial_weights).is_file() or not Path(args.data).is_file():
                raise ValueError("需要本地初始权重/模型配置和真实已标注数据集yaml")
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ValueError('请安装: pip install -e ".[detector]"') from exc
            model = YOLO(args.initial_weights)
            model.train(data=str(Path(args.data).resolve()), epochs=args.epochs, imgsz=640, device=args.device, project=str(Path(args.out).resolve()), name="detector", exist_ok=False, seed=42)
            print("检测器训练完成；请在独立视频核验漏检、反光误检与遮挡后再用于状态分析。")
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
