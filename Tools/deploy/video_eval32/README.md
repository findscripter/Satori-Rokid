# VideoChat3 独立 32 帧采样消融

仅使用已存在、在2026-10-16冻结的 `Research/Data/charades-ego-32frames-v1`。32段各32 JPEG中包含原8帧的字节相同副本，不重新抽样或读其他视频，不传原视频名/标注。GT场景/动作集合与原8帧相同，32帧GT文件的benchmark_id/provenance不同，因此独立使用自己的GT文件事后评分。

本目录是8帧工具的独立快照，只调整帧数量、HTTP parts数量和32帧提示词。`backend32.py`使用同一固定VideoChat3模型、448²像素上限、原生4帧合并、`do_sample_frames=False`；输出4096、greedy、240秒不变。32帧原生8个时序组，8帧原生2组。结果同时改变时间覆盖、视觉token数和计算量，不能将差异纯粹归因为其中一项，也不是新增泛化验证。

原8帧代码、后端、数据与冻结协议均不改；`base-eight-file-sha256.json`记录源快照文件hash。测试含32图片→严格HTTP→旧BenchmarkProvider(frame_count=32)回环、完整EOS与失败计分、真实registry/health冻结检查。`check_processor_cpu.py`只载processor，检查真实32帧tensor及时间，不载模型、不初始化CUDA。

远端独立目录：`/home/gpu-user/work/satori-video-eval/Tools/deploy/video_eval32`；数据：`data32/`，仅manifest/taxonomy/1024 JPEG与匿名registry。GPU服务必须由主代理经gpurun分配，建议新端口18869、独立token/audit/run，不与8帧表混报。正式32段推理前独立freeze，无重试或依据输出换样本。
