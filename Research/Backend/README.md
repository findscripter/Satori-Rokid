# Satori 离线管线

## 2026-10-17 代码修复与方法评分

分阶段方法现在会在发布缓存指导前读取本帧 belief。归一化场景/相关物体签名稳定时复用缓存，变化时重算；不确定观察会延后本次接受，调用者需在可靠观察后重新提交 `accept_step`。`event_outcomes` 记录延后的事件与原因。这是保守的文字签名检查，词语改写也可能触发重算，不等于物体语义跟踪。

尚未人工确认的检查点完成假设可被可靠的 `NOT_STARTED/IN_PROGRESS` 反证撤回；`UNKNOWN/UNRELATED` 或不确定观察不会清除历史假设。新增 `reject_completion` 事件，字段为 `type`、`timestamp_ms`、`step_id`，仅能驳回活动步骤当前的完成提示。驳回后等待新的未完成证据再恢复自动完成提示。人工确认完成仍需 `reopen_step` 才能撤销。

导入作者任务计划时保留检查点具体要求（包括咖啡豆11g等数量）、视觉问题、起始标记和检查提示。整任务物体表作为词汇上下文；原计划未提供逐步物体需求时标记 `required_objects_status=unknown`，不把整任务物体都当作每一步的必要物体。

原 `evaluate` 命令继续评分 Observation 基线。分阶段 `method-timeline.jsonl` 使用独立入口：

```powershell
.\.venv\Scripts\python.exe -m satori_repro evaluate-method --timeline runs\your-method-run\method-timeline.jsonl --run-metadata runs\your-method-run\run.json --labels independently-annotated-method-labels.json --output runs\your-method-score.json
```

标签顶层为 `{"labels":[...]}`。每帧必填 `frame_id`、`active_step_id`（可显式为null）、`phase`、`confirmed_completed_step_ids`。可选：

- `checkpoint_detection`：`step_id`、`overall_status`、`checkpoints:[{checkpoint_id,status}]`，用于独立视觉标签。
- `pending_confirmation`：显式null，或 `{kind,task_id}` / `{kind,step_id}`。
- `guidance_changed`：布尔值，标注该采样记录是否出现新的非空显示指导。

评分分开返回人工输入参与的状态一致率、本帧实际调用的原始检查点检测、确认提示和指导快照变化。缺少可选标签时对应指标为null；缺失预测和标签覆盖率单独报告。指导快照变化不等于墙钟提示时延或任务成功率。当前真实片段的空标签仍不能产生准确率，不会自动用模型答案生成GT。`run.json` 必须声明布尔 `synthetic`，该声明不能证明标注独立性。

修复改变了方法上下文和任务目录；旧失败run只有在完整指纹仍一致时才允许续跑，发生不匹配应新建实验，保留旧证据。

## 独立环境与行为问答

`ask-context` 不需要task文件或任务目录，可直接询问当前环境、对象和行为。默认取所选时间之前最后8帧；它与原分阶段任务指导入口分开。

```powershell
cd C:\path\Satori-Rokid
.\Research\Backend\.venv\Scripts\python.exe -m satori_repro ask-context --session Research\Data\real-task\clean_surface_0_dense_2fps\session.json --window 8 --at-ms 9000 --question "我身处什么环境、正在做什么？" --model YOUR_VISION_MODEL --base-url https://your-endpoint/v1 --output Research\Backend\runs\your-new-context-run
```

凭证来自当前进程 `SATORI_API_KEY`，可用 `--api-key-env`显式更改。命令只调用一次，不自动重试；输出目录必须新建。结果为 `answer.json`、`run.json`、`api-calls.jsonl`、`input-context.json`；原数据集ID只保留本地审计，图片发送前匿名化，不传任务标签或确认事件。当前真实成功样例在 `runs/real-context-8frames-no-imu-20261016`。

可用 `--sensors path/to/imu.json` 追加已有真实IMU回放，文件结构为包含 `samples` 数组的对象。每个样本需完整提供：sample_id、与视频同一时基的timestamp_ms、accelerometer_m_s2三轴数组、gyroscope_rad_s三轴数组、mount（head/phone/wrist/unknown）、source（recorded/live）。原始传感器时钟要先同步和转换；source只是调用者声明，代码不能认证物理真实性。保留图像窗口内最新最多256个样本并记录截断，只把有单位的统计摘要和证据ID给模型，不把加速度范数直接解释成动作或位置。

当前没有真实IMU或音频数据。无IMU时显式标记missing，sensor_evidence_ids必须为空；不得用生成的数字冒充实测。眼镜头动、手部操作、场景类别与绝对位置需分别判断。工程测试281项通过，报告 `Evidence/backend-suite-context`；IMU相关测试使用本地夹具，不能作为融合效果证明。

使用 Python 3.12。当前虚拟环境已准备；重建时运行 `uv sync --extra dev`。不需要眼镜、开发线或API key即可运行合成回归样例。

## 快速运行

```powershell
cd C:\path\Satori-Rokid\Research\Backend
.\.venv\Scripts\python.exe -m satori_repro demo --output runs\my-first-run
.\.venv\Scripts\python.exe -m pytest --cov=satori_repro
```

输出目录必须是新目录，避免覆盖实验结果。包含 `timeline.jsonl`、`run.json`、`summary.json`、`unity-report.json`；demo另有 `evaluation.json`。run记录输入配置哈希，逐帧记录输入图片哈希。失败记录保留并标记failed，不能当完成运行。

## 使用真实视频

```powershell
.\.venv\Scripts\python.exe -m satori_repro extract-video --video C:\path\task.mp4 --output runs\frames-001 --fps 1 --max-frames 600
```

输出JPEG与session.json；抽样时间戳、时间戳回退及截断状态在extraction.json中记录。用户确认事件需要独立写入session的events数组。任务JSON格式参见 `satori_repro/data/demo/task.json`。

视觉接口需显式指定可用的视觉模型，API key从指定环境变量在请求时读取，不写入报告：

```powershell
.\.venv\Scripts\python.exe -m satori_repro infer --session runs\frames-001\session.json --task my-task.json --output runs\prediction-001 --model YOUR_VISION_MODEL --base-url https://api.openai.com/v1 --mode stateful
```

此命令会将所选图片发往显式配置的服务。2026-10-16已用用户授权的远程接口对EgoDex六帧真实图像执行Observation基线，结果在 `runs/real-egodex-baseline-20261016`；该早期基线含描述性帧ID，不用于公平方法比较。自托管接口可更换base-url；普通HTTP只允许loopback。

支持的上下文模式：single只给当前帧；window给最近N帧；stateful给最近N帧和有限过去状态/历史。模型只访问当前时刻以前的输入；评估标签不进入推理。

## 数据与评估

Task包含task_id、title、steps；Frame包含frame_id、timestamp_ms和可选image_path；Event类型为confirm_task、confirm_step、reopen_step。模型Observation字段参见包内demo/responses.json；输出须通过严格验证，证据必须来自实际输入窗口。

用户确认完成与模型观察完成分别保存；重新打开步骤会撤销确认。同帧用户确认切换步骤后，旧步骤帮助文案失效。0.6信心阈值、15秒提示冷却均为待校准工程参数，不是论文原始决策模型。

评估需要独立labels JSON：

```json
{"labels":[{"frame_id":"f0","current_step_id":"cable","decision_kind":"observe","observation_step_id":"cable"}]}
```

`observation_step_id`可省略；其余三个字段必需，步骤可明确为null。

```powershell
.\.venv\Scripts\python.exe -m satori_repro evaluate --timeline runs\prediction-001\timeline.jsonl --labels my-labels.json
```

`state_step_accuracy`包含人工确认/状态更新；`raw_observation_step_accuracy`只依据独立observation_step_id标签，无标签时为null。缺失预测、未标注预测和两种覆盖率都会单独报告。合成数据结果只用于回归检查，不能作为模型性能。

## 与论文的差异

最初CLI `infer` 是Observation工程基线。新模块 `satori_method.py` 单独实现 belief、desire、forecast、checkpoints、guidance 五个阶段，以及目标确认、接受步骤、完成确认、重新打开与拒绝事件；接受步骤不会被记作完成。

在工程根目录运行真实分阶段实验：

```powershell
.\Research\Backend\.venv\Scripts\python.exe Tools\run_satori_api_experiment.py --model YOUR_VISION_MODEL --base-url https://your-endpoint/v1 --output Research\Backend\runs\your-new-run
```

运行前将凭证配置到当前进程 `SATORI_API_KEY`，不要写入脚本。实验使用6帧公开视频、3个候选任务、最多4帧因果窗口；模型输入仅含任务语义和匿名ID，来源映射保存在本地。输出 `method-timeline.jsonl`、`api-calls.jsonl`、`task-catalog.json`、`task-provenance.json`、`frame-id-map.json`、`run.json` 和 `unity-report.json`。每次网络调用记录模型别名、token用量、耗时、输入/提示词哈希和JSON响应，不保存凭证、图片base64或隐藏推理。

最新真实分阶段成功run是 `runs/real-egodex-satori-staged-luna-resume1-20261016`：六帧、15个阶段响应（13新调用+2历史真实响应复用）。失败续跑加 `--resume-from runs/failed-run` 并选新输出目录；只接受模型、端点、原始session/catalog哈希和阶段输入指纹完全一致的成功前缀，首次error之后的记录不复用，不自动重试。`real_api_calls`、`cached_calls`分别计数，缓存的原始usage/latency不能计作本轮新消耗或新耗时。真实实验结果不是由单元测试模拟出来的；233项本地测试使用独立mock验证工程行为。

方法差异：统一VLM替代原CV/BLIP感知组合；使用显式检查点状态及可靠反证回撤，尚未等价验证原fuzzy检测；只支持文字指导；串行预取缓存不代表原论文并行调度的延迟收益；公开视频清洁子任务经过适配。脚本只自动接受模型提出的目标和初始步骤，从不确认完成，因此本次集成实验不覆盖完整任务闭环。无独立真值不报告准确率。原始代码在 `../Reference/satori-assistance`。
