# 数据准备

本目录只保留说明，视频、帧、GT和模型输出需要自行准备；离线demo使用 `Research/Backend/satori_repro/data/demo`，无需下载数据。

| 数据 | 官方或作者来源 | 本仓库入口 |
|---|---|---|
| GTEA | https://ai.stanford.edu/~alireza/GTEA/ | `Tools/prepare_gtea_satori.py` |
| EgoTaskQA | https://github.com/Buzz-Beater/EgoTaskQA | `Tools/prepare_egotaskqa_satori.py`、`Tools/prepare_egotaskqa_frames.py` |
| EgoTaskQA视频作者镜像 | https://huggingface.co/datasets/hyf015/egothinker_eval_dataset | 按准备脚本所需路径放置归档 |
| Charades-Ego | https://prior.allenai.org/projects/charades-ego | `Tools/prepare_charades_gt.py` |

在仓库根目录用后端Python运行对应脚本的 `--help` 查看参数、输入目录和输出格式。数据准备不会启动模型推理；真实推理需另外配置接口或GPU环境。

评测入口：`Tools/satori_paper_benchmark.py`、`Tools/gt_benchmark.py`。对比入口：`Tools/compare_paper_model_runs.py`、`Tools/compare_gt_runs.py`。

本阶段GTEA保留原标注73种动作组合，尚未对齐论文71类及留一被试协议；EgoTaskQA固定200题不是原作者的同一组题号。具体结果和使用范围见 [实验简报](../../docs/experiments.md)。
