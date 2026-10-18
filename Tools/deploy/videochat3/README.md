# VideoChat3-4B on gpu-host

固定模型：`MCG-NJU/VideoChat3-4B`，HF revision
`9f6107bdfee9686dfe493444022f5cf79a1a6acf`。
官方代码仓库核对 revision：`015f52eb2ea70b5abda30192aee18034f2c5f13b`。

来源：[官方模型](https://huggingface.co/MCG-NJU/VideoChat3-4B)、
[官方仓库](https://github.com/MCG-NJU/VideoChat3)。模型及其自定义推理代码为 Apache-2.0。
`official/` 是固定 HF snapshot 中的小型源码/配置参考；运行时使用通过完整性检查的远端 `model/`，不能修改官方权重或源码。

远端个人目录：`/home/gpu-user/work/videochat3-satori`。独立 `.venv` 通过
`--system-site-packages` 复用系统 Torch `2.12.1+cu130`，安装 Transformers
`4.57.6`、Accelerate `1.15.0`、qwen-vl-utils `0.0.14`、decord `0.6.0`、
opencv-python-headless `4.11.0.86`；不安装 XTuner、FlashAttention 或另一套 Torch。
统一 HTTP 服务另用 FastAPI `0.135.1`、Uvicorn `0.42.0`。
官方配置记录开发版 `4.57.0.dev0`，4.57.6 已通过 CPU 模型类导入及完整原生视频预处理。

3 个 BF16 权重文件共 8,944,849,752 字节。`download_model.py` 不使用第二份 HF
cache，每文件核对固定 HF manifest 的 Git blob SHA1 或 LFS SHA256；中断 `.partial`
文件可续传。保留至少 3 GiB 可用磁盘。直连 HF 不通、HF mirror 大文件慢，故权重大文件
可选 ModelScope 镜像固定 commit `9fc3c4a1c5f850399f621332719e0aebea334170`，
3 个大小及 SHA256 均与固定 HF snapshot 一致，小型文件仍用 HF 固定版。
`evidence/download-integrity.json` 的 `complete: true` 才表示全文件已验收。
2026-10-17 已完成 28 文件、8,960,941,495 字节逐文件完整性验收；本地证据在
`Evidence/videochat3-deployment/download-integrity.json`。CPU 原生视频处理、模型空张量
构造（4,472,378,352 参数）及 8 项边界测试通过；GPU 实测由主协调者另行调度。

## 使用边界

- 所有 GPU 计算只能经主协调者 `gpurun` 启动，程序内 `cuda:0`。
- `backend.Backend` 创建时即加载 GPU 模型；CPU 检查仅用 `check_processor_cpu.py`。
- `Backend.generate` 接收统一系统提示、无标签 context、8 张原始 JPEG 解码的 PIL
  与真实原视频 fps/帧序号。将 8 帧作为单个原生 `TxCxHxW` 视频，显式
  `do_sample_frames=False`，不重复补帧、不重新均匀采样。不得捏造 fps。
- 使用官方 SDPA 分支：仅在内存将 `vision_config.attn_impl='sdpa'`，语言模型也用
  `attn_implementation='sdpa'`；原配置/权重文件不变。
- 帧像素预算 56² 到 448²；8 帧按官方 `temporal_merge_size=4` 压缩为两组，
  官方 processor 加入每组首末帧真实时间均值。此视频预处理与多图模型不同，报告须写明。
- 输出只按真实 EOS 判定 `stop`，达到上限或非 EOS 为 `length`；不修 JSON、不重试。
- 操作上下文上限 32,768（输入加预留输出），当前未做长视频容量证明。

`run_proactive.py --model ... --video ... --output 新文件` 使用固定官方
`StreamingSession` 和 `VideoFrameExtractor` 跑实际在线协议；默认 2 fps，每秒一轮，
8 轮，每轮最多 256 输出。Standby 后下一轮按官方规则从 224² 增至 448²。
保存每轮原始输出、EOS/length、时间、token 和实际状态。它属于功能验证，没有事件触发
GT，不能据此宣称主动提醒准确率、实时性或 Satori 任务完成率。

CPU验证入口：

```sh
.venv/bin/python -m unittest test_backend
.venv/bin/python check_processor_cpu.py --model model --output evidence/processor-cpu.json
```
