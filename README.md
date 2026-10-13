> 新增隐藏状态模拟研究：饥饿、紧张、探索、疲劳 → 随机行为 → 视频轨迹 → 三版模型 → 独立模拟测试。运行 `.venv/bin/python -m fishwatch simulate --out outputs/mood_simulation`，打开 `outputs/mood_simulation/index.html`。完整参数与边界见 [模拟状态研究说明](模拟状态研究说明.md)。
>
> 保留 `logistic`（默认）、`gam`、`hmm`，见 [三个版本使用说明](三个版本使用说明.md)。本版扩充行为输入，旧模型需重新分析和训练，已有离线HTML仍可阅读。模拟心理标签禁止用于真实视频；输出概率不是生物学心理测量。

# 小鱼 · 神仙鱼行为与状态建模

## 当前 FishMood 入口（2026-10-10）

产品方向是以一缸已知三条真实神仙鱼为锚点的虚拟陪伴；真实录像目前只支持整缸画面探查，没有可靠的逐鱼身份绑定、轨迹或 H/S/E 标签，不能把花斑、红顶、蓝银的虚拟人格对应到实拍个体。网页站点在 `site/`，含项目主页、算法介绍、真实录像／旧模拟隔离的陪伴 Demo、独立十天和三十天三鱼 synthetic 实验。旧的单鱼四状态视频研究和下方 CLI 说明继续保留，**不是**三十天数据的直接来源。

从项目根目录运行 `python3 -m http.server 8765`，访问 `http://127.0.0.1:8765/site/thirty-days.html`；旧十天实验在 `/site/ten-days.html`。另可解压 `outputs/FishMood-thirty-day-preview-20261008.zip`，进入 `FishMood-30d` 目录启动 HTTP 服务后访问 `/`。ZIP 包含静态预览及生成脚本／测试，不是 Python 环境安装包；不能直接双击 `file://` 预期 ES 模块与数据请求正常工作。详细口径见 [三十天输出说明](outputs/FishMood-thirty-day-results.md) 与 [验证记录](VALIDATION.md)。

30 天实验每鱼连续 720 个小时快照，前十天为预设典型情境；11 项合成动作／三维位置特征用于拟合 Logistic、GAM、有监督 HMM。先用每鱼前 20 天拟合并对后 10 天留出评分，另用全 30 天重拟合参数供展示；两套参数不能互换评分。它们都只识别**当前合成 H/S/E 高低标签**，不预测下次摄食响应，更不估计真实鱼的心理或健康。旧视频管线的 12 项二维候选特征与该 11 项输入不兼容。

本地运行的 Python 项目：**长视频 → 鱼检测/短轨迹 → 行为特征 → 时序状态模型 → 离线 HTML 报告**。

研究对象按淡水神仙鱼 *Pterophyllum scalare* 设计。代码、合成演示、梯度提升树与 TCN 的训练/推理可以运行；项目已有本缸真实录像片段及整缸画面变化探查，但**没有可靠真实逐鱼追踪、独立状态标签或已验证的神仙鱼检测/诊断权重**。原 CLI 演示视频使用合成物体及生成器提供的框，不证明真实鱼检测能力。真实逐鱼模型仍需独立数据和验证。

## 已实现的范围

|部分|实现|
|---|---|
|视频|流式采样，优先解码时间戳，轨迹CSV，采样帧率的标注视频|
|检测入口|本地神仙鱼YOLO权重 / 外部检测框CSV；另有明确受限的运动前景预览|
|轨迹|距离、速度预测、颜色直方图与匈牙利匹配；过期或歧义产生新的短轨迹|
|行为|二维速度、加速度、路径长度、转向率、同向转动一致性、净转圈数、空间分布、邻近距离、群体分散与可见率|
|模型|历史统计 + HistGradientBoosting；或共享因果TCN + 多任务分类头|
|状态任务|温度暴露 low/reference/high；溶氧暴露 low/reference；摄食响应 low/medium/high|
|探索性任务|可显式启用高氧与短/参考/长光周期头；必须有独立标签，光周期至少24小时有效行为历史|
|训练|独立group的train/val/test；缺失标签掩码；验证集温度缩放校准；独立测试及逐组指标|
|拒判|观测质量不足、模型概率不足；合成模型禁止用于真实视频；运动前景禁止用于状态训练/推理|
|报告|离线自包含HTML，指标选择、时间窗滑块、个体观测与缸级概率，数据边界及测试记录|

**尚未实现/验证：**鳃盖或吞食动作识别、永久个体重识别、从真实视频重建三维运动、自动追逐识别、专用多日节律分析、可靠的高氧/光照原因诊断、自动传感器标注、断点续算和真实鱼准确率。`建模方案.md`描述的多尺度、姿态与社会行为部分是后续理想设计，不应与本版本已实现能力混淆。当前TCN使用配置指定的历史并自动扩展感受野；默认是短历史，不是已经训练的多日模型。

## 1. 运行环境

Python 3.10+。交付包不包含 `.venv`；请先按《打包使用说明.md》安装，再使用：

```bash
cd <解压后的项目目录>
.venv/bin/python -m fishwatch --help
```

新机器安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,temporal]"
# 需要真实YOLO检测/检测器训练时再安装：
python -m pip install -e ".[detector]"
```

`requirements-tested.txt`记录本次实际测试的环境版本（不含可选YOLO扩展）。一般优先根据pyproject安装；锁定文件用于复现此环境。YOLO依赖有自己的许可条款，产品分发前按所选依赖许可处理。

## 2. 一条命令跑完整演示

```bash
.venv/bin/python -m fishwatch demo --out outputs/my_demo
```

生成24段一分钟合成视频，12/6/6个独立生成会话分别用于训练/验证/测试。视频→外部框→追踪→特征→真实执行模型拟合→独立测试→报告，全部在本地完成。通常数分钟内完成，取决于设备。

打开 `outputs/my_demo/report.html`。高准确率仅说明模型学到了生成器的人造模式，**不是神仙鱼状态识别的准确率**。每个任务的演示标签都是人为定义，并非生理事实。

复用同一份数据训练TCN，避免重复生成视频：

```bash
.venv/bin/python -m fishwatch train \
  --manifest outputs/my_demo/manifest.json \
  --config outputs/my_demo/config.json \
  --backend tcn --epochs 40 --out outputs/my_demo/model_tcn

.venv/bin/python -m fishwatch predict \
  outputs/my_demo/recordings/test_3_3/analysis/analysis.json \
  --model outputs/my_demo/model_tcn/model.joblib \
  --out outputs/my_demo/tcn_report
```

输出目录已有分析或模型时会拒绝覆盖。换目录即可。`predict`、`report`会更新指定的报告输出。

## 3. 接入真实视频

### 3.1 配置鱼缸

`project_config/config.json`已经生成。新建模板可执行：

```bash
.venv/bin/python -m fishwatch init --out my_config
```

关键参数：

- `tank_roi`：实际鱼缸水体在画面中的矩形，归一化 `[x1,y1,x2,y2]`。侧视时上边界应贴近水面。
- `feeding_roi`：相对**tank_roi内部**的归一化投喂区。
- `expected_fish_count`：已知鱼数；默认null只判断至少一条可见，不能表示全缸覆盖。
- `reference_body_length_px`：可靠的投影体长标定；默认null时速度单位为检测框宽度/秒，不是真实体长/秒。
- `sample_fps`：轨迹采样帧率，默认10；不是呼吸频率测量帧率。
- `window_seconds`：默认10秒；`history_windows`默认6，主模型最长输入约60秒。
- `surface_fraction`、`bottom_fraction`：区域划分用的工程参数，不是生理阈值。
- `decision_threshold`：初始0.7，只是接受预测的工程阈值，必须通过真实验证再确定。

训练和推理必须保持窗口长度、速度单位及区域语义一致。实际鱼缸像素ROI可不同，但位置含义应一致。摄像头固定、侧面拍摄，尽量保留水面与缸底；不要为拍摄移除所有遮蔽物。

### 3.2 有训练好的鱼检测器

```bash
.venv/bin/python -m fishwatch analyze data/tank.mp4 \
  --weights data/angelfish_detector.pt --class-id 0 \
  --config project_config/config.json \
  --out outputs/tank_session_01 --overlay
```

权重必须是本地文件，且类别对应神仙鱼。代码不会自动下载通用COCO模型并假装它能检测鱼。视频识别默认只输出观测；未传入状态模型时不会输出原因概率。

### 3.3 尚无检测器：接外部检测框

```bash
.venv/bin/python -m fishwatch analyze data/tank.mp4 \
  --detections data/detections.csv \
  --config project_config/config.json --out outputs/tank_session_01
```

CSV格式如下。`frame`是原视频从0开始的帧编号；必须排序。坐标是原始图像像素。采样时没有对应检测框的帧被视为未观测，不自动沿用旧框。

```csv
frame,x1,y1,x2,y2,confidence
0,100,60,150,160,0.98
0,220,90,270,190,0.95
1,101,60,151,160,0.98
```

人工标注或外部鱼检测器都可产生此文件。示例数值不是你的真实数据。

仅想检查视频解码和输出格式，可用：

```bash
.venv/bin/python -m fishwatch analyze data/tank.mp4 \
  --motion-preview --out outputs/motion_preview
```

该模式识别的是前景运动物体，会漏掉静止鱼并误检反光、水草，因此禁止拿来训练状态模型或预测状态。

### 3.4 训练神仙鱼检测器

抽帧：

```bash
.venv/bin/python -m fishwatch extract-frames data/tank.mp4 \
  --out data/frames --every-seconds 5
```

用标注工具标鱼体框，导出YOLO格式；划分完整鱼缸/视频，而不是随机拆邻近帧。尽量标一致的躯干范围，避免长鳍改变检测框宽度。目录示例：

```text
fish_dataset/
  images/train/  images/val/  images/test/
  labels/train/  labels/val/  labels/test/
```

修改 `project_config/detection_dataset.yaml` 中的绝对路径，再执行：

```bash
.venv/bin/python -m fishwatch train-detector \
  --data project_config/detection_dataset.yaml \
  --initial-weights data/local_yolo_weights.pt \
  --epochs 60 --device cpu --out outputs/detector_training
```

初始权重也可替换为本地YOLO模型架构yaml。此仓库不附神仙鱼检测权重和训练图片。可选YOLO接口基于官方API实现，当前演示不依赖它。

## 4. 状态模型训练数据

每段视频先生成 `analysis.json`，然后提供同一视频相对时间轴上的独立标签CSV：

```csv
start_s,end_s,task,label,source
0,600,temperature,reference,temperature_logger_protocol_v1
0,600,oxygen,reference,calibrated_DO_logger_protocol_v1
```

上例仅说明格式。不要复制这些标签作为真实真值。`source`必须说明标签来源；它不进入模型特征。

可用类别：

|task|label|解释|
|---|---|---|
|temperature|low / reference / high|根据独立温度记录与预先定义协议标注的暴露，不自动等于热/冷损伤|
|oxygen|low / reference|独立溶氧记录标注，不用浮头行为反推标签|
|feeding_response|low / medium / high|按实际摄食响应协议标注，不是实际饥饿或能量缺口|
|oxygen_high|reference / high|可选高氧暴露研究头，需独立氧饱和度标签|
|photoperiod|short / reference / long|可选光周期研究头，标签来自照明记录，含义是暴露而非已证实不适|

同一任务的标签区间不能重叠。只有完整落在区间内的时间窗才得到该标签；其他标签保持缺失。未标注不是正常。若希望预测投喂前的摄食动机，应把**随后独立测得的响应标签**对齐到投喂前的预测区间，不能让投喂后的画面进入输入。当前代码不会自动推断实验协议或自动生成摄食标签。

温度/DO界限随驯化、物种和实验协议确定，本项目不捏造通用阈值。正常家庭数据不能凭空训练低氧、高低温类别；缺少某类时相应任务会跳过。

训练清单示例（路径相对manifest）：

```json
{
  "recordings": [
    {"analysis":"../outputs/A/analysis.json","labels":"A.csv","group_id":"tank_A","split":"train"},
    {"analysis":"../outputs/B/analysis.json","labels":"B.csv","group_id":"tank_B","split":"val"},
    {"analysis":"../outputs/C/analysis.json","labels":"C.csv","group_id":"tank_C","split":"test"}
  ]
}
```

`group_id`是独立鱼缸/实验批次，不是帧或滑窗编号。单缸按完整日期分组只能证明同缸跨日效果；不能声称跨鱼缸泛化。长连续视频若拆段，边界至少隔开完整历史和目标预测区间，代码不能从不同文件名自动发现所有相邻片段关系。

代码检查同一group、同一路径和相同视频内容哈希跨分组；不会把临近窗口随机分配。任何任务必须在训练和验证集中都有全部类别，才能拟合并校准。默认最少窗口数仅防误用，不是统计学样本量保证。

```bash
.venv/bin/python -m fishwatch train \
  --manifest data/manifest.json --config project_config/config.json \
  --backend tree --out outputs/state_tree

.venv/bin/python -m fishwatch train \
  --manifest data/manifest.json --config project_config/config.json \
  --backend tcn --epochs 60 --out outputs/state_tcn

.venv/bin/python -m fishwatch predict outputs/tank_session_01/analysis.json \
  --model outputs/state_tree/model.joblib --out outputs/tank_prediction
```

只加载自己训练或信任的 `model.joblib` / 检测权重；joblib使用Python序列化，不能安全加载陌生来源文件。

### 可选高氧/光周期研究头

在理想的独立标注数据齐全时，`train --include-exploratory`允许拟合这两个头，默认禁用。训练/验证任一类别缺失则跳过，不伪造类别。低氧与高氧头若同时输出互相矛盾的状态，推理会拒判。

光周期不能靠短片建模。可将一份独立配置设为 `window_seconds: 600`、`history_windows: 144`（24小时），对持续多日的视频重新分析，然后单独训练光周期模型。必须先积累完整24小时有效历史才会得到该任务标签/预测；夜间看不见鱼导致特征缺失时将拒判。窗口与历史长度是工程初值，并不是证明24小时足以可靠区分光照长短。短窗口主模型与长窗口光周期模型需要分别分析、训练和报告，不能混用特征。

```bash
.venv/bin/python -m fishwatch train \
  --manifest data/long_recordings_manifest.json --config project_config/long_history_config.json \
  --backend tcn --include-exploratory --out outputs/photoperiod_research
```

这些接口实现了可训练的研究方案，不意味着已经证明神仙鱼行为可以识别高氧或光照原因。没有证据的任务应继续保持未训练/无法判断。

## 5. 输出解释

```text
outputs/tank_session_01/
  analysis.json      # 窗口特征、个体观测、质量与参数
  tracks.csv         # 采样帧轨迹；无检测帧track_id=-1
  preview.jpg
  tracks.mp4         # 可选，采样帧率；原视频不被修改
  predictions.json  # 未加载模型时状态为observations_only
  report.html       # 可独立拷贝、双击打开，无网络依赖

outputs/state_tree/
  model.joblib
  model_card.json
  evaluation.json   # 完整测试指标与逐group结果
  provenance.json   # 分组和标签来源文件路径
```

模型输出是**缸级**状态概率，个体表格是直接观测。报告中的“摄食响应”不能直接改称“鱼饿了”；“上层停留”不能直接改称“浮头喘气”。低模型概率和低质量会拒判，但现有门控不能保证发现所有未知分布，高置信度仍可能错。

内存中仅保留当前窗口的逐帧观测及各窗口统计，不保留整段视频；解码仍需遍历源视频。视频若为变帧率，应核对时间戳；出现时间戳回退提示时建议先转恒定帧率。当前不支持断点恢复，遇到错误请使用新输出目录重跑。

## 6. 验证

```bash
.venv/bin/python -m pytest -q
# 三十天实验测试文件使用 .test.py 命名，需显式调用：
.venv/bin/python -m pytest -q --import-mode=importlib webui/tests/thirty_day.test.py
```

默认发现测试与三十天定向测试是两套入口；最近记录分别为 30 passed 与 4 passed，详见 [VALIDATION.md](VALIDATION.md)。测试覆盖：物理速度与缺口、漏检不等于静止、已知鱼数覆盖率、历史因果性、轨迹过期、重叠标签、跨组泄漏、合成/真实域隔离、HTML注入转义、真实视频解码与标注输出、TCN因果性。

## 7. 研究与实现依据

- [神仙鱼温度变化视频与LSTM研究](https://doi.org/10.5657/KFAS.2021.0965)
- [神仙鱼社会等级与摄食/活动](https://pubmed.ncbi.nlm.nih.gov/12689421/)
- [神仙鱼时间—地点学习](https://pubmed.ncbi.nlm.nih.gov/16129239/)
- [TCN原作者实现](https://github.com/locuslab/TCN)
- [Ultralytics检测预测API](https://docs.ultralytics.com/modes/predict/)

当前跟踪器是本项目的保守短轨迹基线，**不是ByteTrack或DeepLabCut的实现**。后续可在独立真实数据上比较这些方法。
