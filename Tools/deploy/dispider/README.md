# Dispider / Satori 部署

2026-10-17，当前目录记录部署工具与 CPU 验证，实际 GPU 验证由主协调通过 `gpurun` 执行，结果另存。

已完成固定revision全部24文件、19,511,273,821字节下载，4片均通过官方LFS SHA256。随后实际运行CPU恢复/热缓存复核，24文件官方哈希均通过，见 `evidence/final-download-integrity.json`。第3/4片采用16路CDN续传，复用与PC完整原片核对过的573,440,000字节上传前缀，约812.5秒完成；`.upload`与`.incomplete`均已并入最终文件，不留冗余。临时公开CDN能力文件与小型Range探测文件已按精确白名单清理，hash记录保留。压缩归档完整性仍应独立检查下述final archive记录。

| 项目 | 固定值 |
| --- | --- |
| 官方源码 | https://github.com/Mark12Ding/Dispider ，`803f6b4634b3232293e89a38bcce9f5815c33ed4` |
| 官方权重 | `Mar2Ding/Dispider`，`7c5517eaaa4dc3788602af40e077063792c75f6c` |
| 服务器目录 | `/home/gpu-user/work/dispider-satori` |
| 工作权重 | 私有 RAM 目录 `/dev/shm/gpu-host-satori-dispider-7c5517e` |
| 原权重大小 | 4片共19,501,019,240字节；20个元数据文件10,254,581字节 |
| 模型结构 | Qwen2-7B Reaction + Qwen2-1.5B Perception/Decision + CLIP，复合权重各存一次 |
| 依赖 | 项目 `.venv`，复用现有 Torch2.12.1+cu130，Transformers4.41.2、Accelerate0.27.2、Decord0.6.0 |

官方验证栈为 Python3.10 / Torch2.2.0+cu121 / FlashAttention2.5.9.post1；本部署采用 Python3.12 和 SDPA。官方源码强制 Decision 使用 FlashAttention2，所以 `prepare_runtime.py` 保留未改的 upstream 副本，另建 runtime，仅允许通过环境变量选择 SDPA。补丁及原文件、改后文件 SHA 在 `evidence/`。权重不改、不量化；CPU 随机小模型检查不能证明完整 FlashAttention 模型等价。

CPU 实测：85个官方测试通过；8项额外检查通过，包括 eager/SDPA 输出、左右 padding、未来 token 不影响前缀、增量 KV 与整段一致、非法 KV 位置被拒。最大输出差为4.77e-7。详见 `evidence/upstream-cpu-tests.xml` 和 `evidence/sdpa-cpu-check.json`。

恢复工具另有20个CPU检查通过，覆盖完整24文件恢复/热缓存复核、坏SHA、损坏zstd、解压超大小、空间不足、私有权限、下载与实际`.upload`上传互斥、路径遍历和symlink逃逸。见 `evidence/restore-cpu-tests.xml`。这些不涉及GPU，也不替代模型效果评估。

独立环境中还复用冻结HTTP服务的20个CPU协议检查，含鉴权、截断、无JSON修复及原BenchmarkProvider真实报文往返；见 `evidence/http-protocol-cpu-tests.xml`。与最新20个恢复检查共40个已通过，两个XML分别保留；此前合并执行39项的历史XML也保留。公共冻结工具没有修改，测试读取独立副本。模型运行依赖在 `requirements-cpu.txt`，测试额外依赖在 `requirements-test.txt`；实际venv精确列表另存远端 `evidence/requirements-local-frozen.txt`。

## 下载与持久保存

远端直连 HF 失败，本机可达；20个小文件由本机拉取并验证 Git blob SHA 后上传。权重使用 `hf-mirror.com` 固定 revision 路径，最终必须匹配官方 HF LFS SHA256，不能仅凭镜像文件名认定来源一致。

`download_ranges.py` 采用8路32MiB Range、单一顺序 append writer。强制 HTTP206、精确 Content-Range 和实际字节数，每段 SHA 和连续完成位置写原子 ledger。恢复时复核已完成段，最终再验证整片官方 SHA。没有重复 HF cache 或预分配完整权重的 RAM 洞文件。

原 BF16 权重从 RAM 加载。可将每片无损 zstd level3 压缩到 `model-archive`，逐片解压 SHA 验证；这不是量化。4片首4MiB抽样预测 archive 14,905,146,909 字节，+5%为15,650,404,255字节，真实压缩率仍需逐片核验。写 archive 前扣除其他模型下载预算并保留至少3GiB磁盘；不满足则保留 RAM 文件并跳过该片归档，不能挤满共享磁盘。RAM 内容重启后会丢失。

VideoChat3 完成下载后，Dispider 的归档余量可以固定保留3GiB。所有原片完成官方SHA核验后，使用以下CPU命令补齐归档；已有片会先解压核验SHA再复用，不重写它们：

```bash
cd /home/gpu-user/work/dispider-satori
.venv/bin/python archive_remaining.py \
  --metadata model-metadata.json \
  --root /dev/shm/gpu-host-satori-dispider-7c5517e \
  --archive-root model-archive \
  --output evidence/archive-integrity-final.json \
  --reserve-bytes 3221225472
```

永久恢复资料是 `model-metadata.json`（固定HF revision和官方文件大小/哈希）、`model-metadata/`（20个小文件）、`model-archive/`（4个zstd）。`evidence/archive-integrity-final.json` 只有全部4片完成无损核验才为 `status=completed`。不要仅凭下载进程结束或片名存在认定归档完成。

实际归档已由主协调的唯一CPU进程完成：4片均 `verified_lossless`，共15,198,663,318压缩字节，逐片解压SHA与官方原片一致，最终状态 `completed`，保留3GiB预算。此记录已同步到本地 `evidence/archive-integrity-final.json`。

## 冷启动、热缓存和停服

`restore_model.py` 是纯CPU、离线工具。它只向固定私有 `0700` 目录 `/dev/shm/gpu-host-satori-dispider-7c5517e` 写24个白名单文件，拒绝路径遍历、symlink、错误所有者和空间不足；预留512MiB RAM文件系统余量。存在下载/上传 partial 时立即拒绝，避免干扰进行中的下载。现存文件逐一重算官方哈希，缺片才从zstd解压到临时文件，原SHA通过后原子落盘。元数据按官方Git blob SHA1验证，4片按官方LFS SHA256验证。每次生成独立恢复audit和RAM中的 `restored-integrity.json`。

```bash
# 独立CPU恢复/核验，不占GPU，不联网。重启后也用同一条命令。
cd /home/gpu-user/work/dispider-satori
.venv/bin/python restore_model.py

# 单卡队列启动：自动先恢复缺失RAM片并复核全24文件，再载入模型。
gpurun -n 1 -d bash /home/gpu-user/work/dispider-satori/run_server.sh \
  --token-file /home/gpu-user/work/dispider-satori/.secrets/api.token

# 查队列实际作业号，确认属于自己的Dispider服务后再取消。
gpuq
gpucancel <Dispider作业号>
```

token由外部私有文件提供，当前用户所有、权限0600、绝对路径且不接受symlink；脚本不写入或打印内容。`run_server.sh` 固定 model `Dispider`、HF revision、端口 `127.0.0.1:18868`、共用 `registry-all.json`。不接受命令行覆盖它们。它保留gpurun的物理GPU映射，backend只用映射后的逻辑 `cuda:0`，没有擅自设置物理GPU0。

HTTP audit每次启动独立存为 `evidence/http-audit-<UTC时间>-<job>-<PID>.jsonl`，启动分配记录同目录 `server-start-*.json`。CPU恢复audit为 `restore-*.json`。服务停止不删除RAM热缓存、原压缩归档或任何结果。系统重启后RAM丢失，永久资料足够离线重建；无需手工拼接权重。该脚本不创建开机服务。

## 两种验证入口

首次真实GPU作业425已完成原生streaming smoke：10秒非GT EgoDex视频产生1个尾窗（16槽、11个独立采样时间，最大已供模型时间9.966666秒），Decision分数-0.65576171875，缓存与oracle分数接近，未触发Reaction、有效回答0。`native-proactive-001.json`及exit记录保留；此结果只证明链路运行和该次沉默行为，不构成检测准确率。

首次非GT分类schema001自然EOS但生成3个token并发出`<keep_silent>`，按原语义抑制为空文本，严格接口返回502。审查发现适配器把taxonomy/context和帧标签传给PerceptionDecision时漏传原system中的实际分类任务；Reaction已正确收到system。GT正式推理尚未开始。现仅将PD调用改为 `preprocess_question([system_prompt + "\n\n" + question], ...)`，Reaction prompt/user原样。旧backend字节及SHA、单行diff保留；新SHA为 `1638dce40f625dd9a03c429339a3f1fa01876b8365106b8cda531df15474d1d9`。

`check_pd_routing_cpu.py` 用真实tokenizer/processor及8张合成图运行旧/新adapter，在实际模型生成前截获，两种image-tag配置均通过：PD确实收到原system+原用户数据，Reaction完整prompt字节、输入token、图像、时间序列、分隔token和预算完全相同。没有读取GT、加载模型权重或执行GPU生成。见 `evidence/pd-routing-fix-cpu-20261017.json`、`.patch` 与旧 `backend-before-pd-routing-20261017.py`。该修复不关闭silence、不强制回答，不改官方native query流程；后续效果须由新的schema/正式记录确认，旧schema001仍保留。

`backend.py:Backend` 对接通用 `Tools/deploy/video_eval` loopback 服务，使用官方 offline Reaction 生成流程，保留 Perception/Decision 记忆提取。它每次用户提问调用一次 Reaction，不测原生 streaming Decision 的主动触发闸。`<keep_silent>` 仍按官方语义返回空回答，由评测保留为失败或弃答。

冻结分类仅用8张独立 JPEG；每张相邻重复2次以满足16帧 clip 输入，总信息仍为8张。保留官方 nearest square resize、图片处理器和模型 conversation 结构，system/user 文字使用同一分类协议；所有预处理、真实首次 Reaction decoder `inputs_embeds` 长度与 EOS/length 都返回给服务。没有 JSON 修复、自动重试或额外视频帧。

`run_proactive.py` 使用官方 streaming adapter，默认16秒窗口、每窗16个均匀采样帧，源视频不强制改为2fps。它记录每窗真实采样时间、尾窗重复情况、所供模型的最大时间戳、Decision 分数与触发、KV cache/oracle、实际 Reaction 文本和 EOS/length。CPU decoder 会预读；所供模型的最大采样时间与 CPU 解码预读明确区分。该 smoke 无独立事件 GT，不报告准确率。

原生streaming保留官方react解码行为，原始输出另存；即使`skip_special_tokens`删去`<keep_silent>`后仍剩文字，也标记`keep_silent_emitted=true`、`effective_answer=false`，不把它算有效答复。`emitted_event_count`是原入口实际发出的事件数，`effective_answer_count`还要求无silence、非空和真实EOS。

另有独立 `run_offline_smoke.py`，在正式GT分类结束后由主协调另起gpurun，只调用一次官方 `inference.VideoStream.run`。固定自然query为 `Describe the visible environment and what the person is doing.`，同一10秒非GT EgoDex视频，唯一生成覆盖是把官方1024输出预算设为请求的256；没有JSON合约、候选标签、重试或关闭silence，也不改backend/runtime/frozen文件。它观察官方实际`VideoReader.get_batch`请求，记录真实采样索引/时间、原视频SHA、全部新token、原始特殊token、EOS、官方visible返回值与silence。3个纯CPU记账检查通过，验证正常EOS、silence伴随非空文字以及截断的记录；不是视频模型效果测试。

```bash
# 必须在独立gpurun单卡分配中执行；单次输出路径不可覆盖。
export DISPIDER_RUNTIME=/home/gpu-user/work/dispider-satori/runtime
/home/gpu-user/work/dispider-satori/.venv/bin/python \
  /home/gpu-user/work/dispider-satori/run_offline_smoke.py \
  --output /home/gpu-user/work/dispider-satori/evidence/offline-freeform-001.json
```

必须在 `gpurun` 分配中设置 `DISPIDER_RUNTIME=/home/gpu-user/work/dispider-satori/runtime`，使用该项目 `.venv/bin/python`。`Backend` 启动时验证 `GPURUN_JOB` 与 `CUDA_VISIBLE_DEVICES`，不允许绕过队列。

```bash
# 仅由主协调在 gpurun 分配中执行，权重完整 SHA 验证后再启动。
export DISPIDER_RUNTIME=/home/gpu-user/work/dispider-satori/runtime
/home/gpu-user/work/dispider-satori/.venv/bin/python \
  /home/gpu-user/work/dispider-satori/run_proactive.py \
  --model /dev/shm/gpu-host-satori-dispider-7c5517e \
  --video /home/gpu-user/work/satori-video-eval/smoke/native-smoke.mp4 \
  --output /home/gpu-user/work/dispider-satori/evidence/native-proactive-001.json
```
