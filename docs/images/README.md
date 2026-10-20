# 图片说明

两张截图均为2026年10月16日项目在Unity 2022.3.62f3c1中的原始运行截图，原分辨率1440×940。`unity-mage-context.png` 裁掉了最底部一行本机运行路径，现为1440×922；其余内容均未修改界面、模型回答或数值。`forecast-horizon.png` 是根据本项目实验结果绘制的图表，不是截图。

| 图片 | 展示内容 | 实验范围 |
|---|---|---|
| `unity-task-replay.png` | Satori任务状态、当前观察、文字指导、完成状态和回放时间轴 | 真实模型记录的离线回放；6个采样时刻中的第3帧，没有用户完成确认 |
| `unity-mage-context.png` | Mage-VL八帧输入、环境与行为问答、原始JSON | 同一视频5.5至9.0秒窗口的一次真实请求；端到端16.68秒，无IMU |
| `forecast-horizon.png` | 三个API模型在0、1、3秒前瞻下的动作预测准确率 | 七模型主实验的GTEA动作预测，每个点为答对题数/32 |

## 视频素材来源

截图内的视频画面来自 **EgoDex** 的 `test/clean_surface/0.mp4`，由Apple公开发布，用于说明本项目的第一视角研究界面。

- 数据集作者：Ryan Hoque、Peide Huang、David J. Yoon、Mouli Sivapurapu、Jian Zhang。
- 官方项目：https://github.com/apple-aiml-research/ml-egodex
- 论文：https://arxiv.org/abs/2505.11709
- 数据许可入口：https://github.com/apple-aiml-research/ml-egodex#license

图片展示少量视频画面及本项目界面，不提供原始视频或完整数据集。第三方视频素材的权利与许可归原作者；本项目的代码许可说明不改变其适用条件。

## 结果解释

图中回答与置信值均为当次模型输出，可能包含错误或未证实的细节；截图只展示功能与运行状态，不代表模型准确率、完整任务成功率或眼镜实测结果。任务回放图中的指导引用其生成时的历史帧，当前观察帧与指导生成证据分别保留。
