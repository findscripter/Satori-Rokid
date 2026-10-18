# VideoChat3 / Dispider 受控输入协议

主比较只复用已经冻结的 Charades-Ego 32 段、每段 8 个 JPEG。模型不读取原片、GT、原视频名、脚本、对象表、其他样本答案。候选固定全部 16 场景和 157 动作。系统提示、可见 context JSON、帧 ID 和真实毫秒时间与既有 Mage/GPT 一致。两模型 native template/视频 token 和视觉预处理有固有差异，必须报告，不能声称像素 token 或模型计算相同。

- VideoChat3：8 个 JPEG 解码为 uint8 视频 tensor；传实际 decoder fps、原始帧索引和原片帧数，`do_sample_frames=False`，原生每 4 帧时序合并。
- Dispider：8 个 JPEG 各连续重复 2 次成为唯一 16-slot clip，槽映射 `[0,0,1,1,...,7,7]`。信息预算仍为 8 个独立帧，不能称为新增了 8 帧观测。记录其原生方形缩放/CLIP 处理和槽映射。
- `prepare_metadata.py` 仅从帧 provenance 提取无标签 decoder 数值，以 JPEG SHA + 时间序列指纹建表；不把 provenance 的 source_video_id 交给 backend。
- 每模型每样本一次调用，确定性解码，最多输出 4096 token，不自动重试、不去 Markdown、不修 JSON。仅自然完整 EOS 且严格四键 JSON 才为有效分类。完整但内容不确定允许原生 null 弃答。
- HTTP 失败、截断、格式失败、弃答、未运行全部按旧 scorer 保留 32 样本分母；另可报共同有效样本，但不能替代主结果。失败原始文本仅在私有服务 audit 保存，客户端错误体只带 code / finish_reason / usage。
- 旧 Mage/GPT 是经 SSH/公网的 HTTP 延迟；本轮 CPU loopback 请求延迟不能作纯算力或相同网络性能比较。另保存服务端 generate 耗时、token usage（无法真实计数则缺失，不虚构）。
- 正式推理前冻结 manifest/taxonomy/prompt/GT/registry/代码 SHA 与模型 revisions；`run_cpu_benchmark.py` 在本机远程服务旁发 CPU HTTP 请求，凭据只进临时子进程环境，事后独立 GT 评分。

本轮原生在线模式只用非 Charades 评估集的 EgoDex 样本验证标签/触发时间、因果输入和持续状态，不据整片 Charades 动作 ID 集合构造每时刻帮助真值。Charades 本地 376 个时间区间有 54 个不满足 `0 <= start < end <= decoded_duration`，涉及 15/32 段；因此未经重新冻结合法区间评分协议，不发布在线时机准确率。即使将来用合法官方动作区间评分，也仅代表动作发生检测，不能代表用户需要帮助的时间。
