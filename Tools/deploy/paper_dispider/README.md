# Dispider 四帧公开源数据适配

本目录为 2026-10-18 新增的独立适配器。复用固定的 Dispider 权重
`7c5517eaaa4dc3788602af40e077063792c75f6c` 和官方源码
`803f6b4634b3232293e89a38bcce9f5815c33ed4` 的已记录 SDPA 运行时。
旧 `Tools/deploy/dispider/backend.py`、共同评分器与所有既有结果均未改动。

每次只读取冻结 manifest 的四张原始 JPEG、原时间戳和完整共享候选。
四张图分别相邻重复四次，占模型要求的 16 个内部槽位；这仍然只有四个唯一观察。
保留官方 nearest square resize、CLIP processor、PD 记忆提取和一次离线 Reaction。
不测试原生流式 Decision 触发；没有 GT、音频、IMU 或额外视频帧进入模型。
所有输入映射、时间戳、处理器形状、原始含特殊符号文本和生成 token 均保存在 trace。

推理使用共同 `satori_paper_benchmark.PROMPT`、4096 输出预算、greedy，一次/样本。
`<keep_silent>` 保留为原生沉默并记协议失败；自然语言输出也按原 JSON 合约记失败，
不抽取文字标签、修 JSON、补问、重试、调整门限或更换样本。
因此零分表示此合约下没有交付合格答案，不能独立推导视觉能力为零。

远端目录 `/home/gpu-user/work/satori-dispider-paper-20261018` 由已经校验的纯 infer
归档建立，推理前扫描没有 `gt.json`，完成所有正式推理后才引入评分标签。
`run_preflight.sh` / `run_suite.sh` 必须用 `gpurun -n 1 -d bash ...` 调用，
程序只访问逻辑 `cuda:0`；不会创建常驻服务。所有 output 拒绝覆盖。
`prepare_remote.py` 和 launcher 是本次固定实验的可审查执行记录，重复实验应创建新目录、协议和 run。

`provider.py` 保留同一 PROMPT/payload 的 SHA 和图像字节 SHA，
`audit_native.py` 另核对 4→16 映射、时间、原始 silence 与 token 计数，
`finalize_remote.py` 在评分前检查完整 324 次推理与已冻结代码 SHA。
`check_runtime_identity.py` 比较当前 90 个上游/运行时 Python 文件和历史固定源字节。

本轮三个 dataset protocol 在首调前已经冻结模型/参数/输入/代码。
总比较协议文档先于正式调用创建，但其 SHA 关联 sidecar 在 job442 首调之后才补写；
这一时序差异在 sidecar 中明确记录，不回写原 protocol，也不声称关联文件是预注册件。
另有一项记录缺陷：纯 infer 包没有 data/protocol.json，本次 `prepare_remote.py`
因此未将 `source_gt_sha256` 写入模型协议。保留该执行脚本与冻结协议原样；本目录脚本
是本次执行证据，不应直接作为新实验的通用准备器。实际评分标签已经由标准库独立脚本
对照历史 dataset/protocol.json 和先前固定 GT SHA 验证一致，相关结果明确标注事后核验。

CPU 合约测试：`python -m pytest Tools/deploy/paper_dispider/test_provider.py -q`。
真实结果与主控比较报告位于 `Evidence/satori-all-models-20261018/`。
