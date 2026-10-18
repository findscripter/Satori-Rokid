# Mage-VL / Satori 部署

2026-10-16。用户已确认部署目标为SSH别名 `gpu-host`，在个人工作区新建目录。

| 项目 | 实际配置 |
| --- | --- |
| 服务器项目 | `/home/gpu-user/work/mage-vl-satori` |
| 公开权重 | `/srv/team/models/Mage-VL`，项目内 `model` 为符号链接 |
| 模型来源 | `microsoft/Mage-VL`，HF固定版本 `d88b153285f1633a61b2f693c59c8576693af185` |
| 下载 | ModelScope微软镜像；关键源码与两片权重对照HF固定版本，22文件共9,499,794,116字节均通过SHA256 |
| 环境 | 项目 `.venv`，复用 `/opt/conda` CUDA PyTorch 2.12.1+cu130；Transformers 5.7.0，BF16，SDPA |
| GPU | 经 `gpurun -n 1` 分配；当前物理GPU 2，程序内为逻辑cuda:0 |
| 服务 | 独立FastAPI适配器，`127.0.0.1:18866`，模型ID `Mage-VL` |
| 本机入口 | SSH转发后的 `http://127.0.0.1:18866/v1` |
| 鉴权 | 随机Bearer凭据，服务器 `.secrets/api.token`，权限600；不写入源码、日志或本地配置 |
| 当前服务总预算 | **96,000 token，输入+预留输出合计**；默认输出4,096，上限8,192 |
| 其他准入限制 | 最多128图、2,000,000文字字符、64MiB请求；实际token在处理器展开后核验 |

2026-10-16已按用户要求尝试扩大上下文：固定权重声明262,144文本位置，tokenizer声明131,072，没有原生1M配置；A100 40GB上单图合成容量测试曾返回96,458输入，106,176/114,368/130,752输入均发生GPU OOM。最终启动档位采用96,000总预算，并在新作业上实际返回95,680输入；它不是精确硬件极限或长视频准确率保证。短输入标记复述通过，95,680重复文本标记复述未通过。完整验收见本机 `Evidence/MAGE-CONTEXT-CAPACITY-20261016.md`。不要把输出8192上限或RoPE频率误当上下文窗口；没有开启原生1M或1M外部记忆。

## 本机使用

在Windows PowerShell运行：

```powershell
& C:\path\Satori-Rokid\Tools\Ask-MageVL.ps1 -CheckOnly
& C:\path\Satori-Rokid\Tools\Ask-MageVL.ps1
```

脚本会检查/建立隐藏SSH隧道，直接通过SSH把凭据捕获到进程内存，然后执行现有 `ask-context` CLI，退出时恢复环境变量。默认输入公开EgoDex视频的连续8帧（2fps，5.5–9秒）；可用 `-Question`、`-Session`、`-AtMs`、`-Window` 和 `-Output` 改配置。新输出目录不得覆盖已有实验。

HTTP支持 `/health`、`/v1/models`、`/v1/chat/completions`，均需鉴权。当前接口接受图片data URL与文字，只支持单请求非流式生成；不接受外部图片URL或本地文件路径。JSON模式是提示词加完整JSON校验，非约束解码；非法/截断输出返回502，不修补成伪成功。原生视频通过下面的独立CLI验证，HTTP元信息的 `native_video_supported=false` 是准确的。

## 服务器启停

先查看 `ssh gpu-host gpuq`。本次容量配置验收后服务作业为 **419**，最新分配以项目 `evidence/scheduler.json` 和 gpuq 为准。如服务已退出：

```powershell
ssh gpu-host 'gpurun -n 1 -d bash /home/gpu-user/work/mage-vl-satori/run_server.sh'
```

服务会持续占用一张GPU。停止时先用gpuq核对自己的服务作业号，然后 `ssh gpu-host 'gpucancel <作业号>'`。不绕开队列或固定物理GPU号。服务器重启后需要重新提交该启动命令；本部署没有修改系统级开机服务。

本机隧道记录在 `Evidence/mage-vl-tunnel.json`；关闭本机转发可结束该记录中对应的SSH进程，先核对PID和命令行，避免PID重用。关闭隧道不停止远端模型。

本机隧道由 `Tools/start_mage_tunnel.py` 显式请求breakaway，并验证子进程已脱离启动器当前Windows Job；`in_any_job`另外保留，不宣称脱离所有Job。每次启动独立日志，已有监听的进程身份校验保持。真实测试仍发生过远端`Connection reset`，所以这项改进不保证网络不中断。大文本容量验收改用服务器本机CPU HTTP客户端 `probe_context_remote.py`，GPU工作始终留在gpurun服务；原网络失败记录保留。

## 真实验证

- 原生frames视频：`evidence/native-frames-002.json`，正常EOS，17个输出token，生成约1.42秒；冷启动全流程约9.18秒。
- 原生codec视频：`evidence/native-codec-002.json`，正常EOS，17个输出token，生成约1.50秒；全流程约14.63秒，含codec预处理/模型加载。
- HTTP环境问答：本地 `Research/Backend/runs/mage-vl-context-server-final-20261016`，8帧，1次真实调用，2232输入/188输出token，完整JSON；服务重启后已通过用户入口再次验证。
- 健康/鉴权：正确凭据ready，无凭据401；服务只监听服务器loopback，通过SSH访问。
- 工程测试：325 passed（281后端+44部署工具），详见 `Evidence/backend-mage-deployment-tests`。这些使用本地mock；真实推理记录单独保存。

最早一次宽泛视频描述产生重复并达到token上限，保留为 `native-frames-001.json`；简短问法自然结束，没有修改权重或强行截断后标成功。输出中“喷洒”等行为尚无独立标签核验，部署成功不等于准确率评估通过。

## Codec 输入格式

原EgoDex MP4实际是MPEG-4 Simple Profile，提取器无法提供bitcost；此前 `native-codec-001.json`正确记录失败。另存H.264 lossless（libx264/crf0）后，300帧的解码MD5逐帧完全一致；见 `evidence/h264-transcode-proof.json`。原视频保留，转换仅用于私有研究验证。codec处理器生成8个canvas，patch位置覆盖64个源帧坐标，不是捏造运动数据。诊断用FFmpeg wrapper未作为修复使用，未改动模型或venv源码。

原生测试通过 `native_video_smoke.py` 运行，必须在gpurun分配的一张GPU上执行；codec需要PATH包含项目 `.venv/bin` 的 `cv-preinfer` 及系统ffmpeg。脚本支持frames/codec，保留实际视频时间和patch位置；新的结果路径不能覆盖已有结果。

## 范围

基座的图像、多帧、原生frames及传统codec路径已验证。主动流式gate、神经codec、实际IMU融合没有启用；当前未安装Mamba/FlashAttention或下载额外gate/DCVC权重。Unity仍使用此前任务回放，新服务已接通Python环境/行为问答入口。后续模型对照需统一采样、像素预算和独立标签，不能直接比较两个接口的token或单样本描述推断准确率。
