# Satori-Rokid

基于 Satori 思路的第一视角环境与任务理解研究原型，包含 **Unity 回放客户端、Python 方法后端、公开数据评测工具**。

当前已实现多帧观察、目标提议、下一步预测、进度检查、文字指导与缓存。工程正在向 Rokid 场景推进，尚未完成眼镜实时相机、语音和 IMU 接入，也尚未验证完整任务交互效果。

## 实验结果

已对七个模型完成 GTEA 目标识别28题、未来动作预测96题、EgoTaskQA问答200题的模块对比，累计2268个模型—样本结果。

- Claude Opus 5.5：目标识别 **89.29%**，动作预测 **55.21%**，问答词法匹配 **22.50%**。
- GPT-6-astra：目标识别 **67.86%**，动作预测 **60.42%**，问答词法匹配 **19.50%**。
- GPT-6.1-sol：目标识别 **89.29%**，动作预测 **54.17%**，问答词法匹配 **17.50%**。

完整模型表、输入设置、失败说明和当前结论见 **[实验简报](docs/experiments.md)**。失败与弃答保留在固定分母；各模型使用相同观察，但推理配置并未完全统一。该评测不是完整 Satori 系统复现或官方榜单结果。

## 环境

- Unity：开发与验证版本为 **2022.3.62f3c1**，记录在 `ProjectSettings/ProjectVersion.txt`。
- Python：**3.12 或以上**；依赖锁定在 `Research/Backend/uv.lock`。
- 离线演示不需要眼镜、GPU或模型密钥。
- 本仓库包含 MCP for Unity 编辑器包。模型权重、公开视频、实验原始响应和机器配置需另外准备。

## 运行后端离线演示

安装 Python 与 uv 后，在仓库根目录执行：

```powershell
cd Research/Backend
uv sync --extra dev
uv run python -m satori_repro demo --output runs/demo-001
```

首次安装依赖需要网络；演示本身使用仓库内的合成数据，不调用模型。输出目录必须是新目录。生成的 `unity-report.json` 可供 Unity 加载，`timeline.jsonl` 保存逐帧状态。

后端测试：

```powershell
uv run python -m pytest
```

完整代码与评测工具测试可在仓库根目录运行：

```powershell
.\Research\Backend\.venv\Scripts\python.exe -m pytest
```

部分部署适配器测试需要各自目录中的额外依赖；数据集检查在数据未下载时跳过。测试和合成演示结果不作为模型准确率。

## 运行 Unity 回放

1. 在 Unity Hub 中打开这个仓库目录。
2. 打开 `Assets/Satori/Scenes/SatoriReplay.unity`，点击 Play。
3. 默认显示内置合成回放，可播放、暂停、调整速度和拖动时间轴。
4. 查看自己的后端输出时，在场景的 `ReplayHud` 组件中填写 `reportPathOverride`，指向生成的 `unity-report.json`；或使用构建程序的 `-reportPath` 参数。

`Satori/Open Mage-VL Test` 是另一个环境问答入口，需要先自行部署和配置模型服务。它不等于完整的任务确认与步骤推进闭环。

如需 Unity MCP，可运行 `Tools/Start-UnityMcp.ps1`，再使用 `Satori/Connect Unity MCP` 菜单。服务默认仅监听本机 `127.0.0.1:8080`。

## 连接视觉模型

后端支持 OpenAI-compatible 视觉接口。密钥只从进程环境变量读取；配置 `SATORI_API_KEY` 后，显式提供模型与接口地址。仓库中的 `.env.example` 仅说明变量，不会被程序自动加载。

```powershell
cd Research/Backend
uv run python -m satori_repro extract-video --video C:/path/task.mp4 --output runs/frames-001 --fps 1 --max-frames 600
uv run python -m satori_repro ask-context --session runs/frames-001/session.json --window 8 --at-ms 9000 --question "我身处什么环境，正在做什么？" --model YOUR_VISION_MODEL --base-url https://your-endpoint.example/v1 --output runs/context-001
```

真实推理会将所选图像发送到你指定的模型服务。部署脚本中的 `gpu-host`、`gpu-user` 和示例路径需按自己的环境配置；这些是部署模板，不会提供原开发机器的服务器或账号。

## 目录

| 路径 | 内容 |
|---|---|
| `Assets/Satori` | Unity场景、回放UI、环境问答与检查工具 |
| `Research/Backend` | 分阶段方法、模型接口、状态机、CLI与测试 |
| `Tools` | 数据准备、模型评测、评分与对比脚本 |
| `Tools/deploy` | Mage-VL、VideoChat3、Dispider与CLI适配工具 |
| `Research/Data/README.md` | 数据准备入口与公开来源 |
| `Research/Reference/satori-assistance` | 带原MIT许可的任务计划参考文件 |
| `docs/experiments.md` | 简版实验报告 |

## 数据与第三方来源

数据集视频、模型权重、原始实验响应、构建产物和运行缓存不包含在仓库中。获取数据后需按相应数据集条件使用，不能把合成演示当作真实评测。

研究参考：[Satori 作者代码](https://github.com/VIDA-NYU/satori-assistance)。第三方代码的来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本项目是独立研究实现。
