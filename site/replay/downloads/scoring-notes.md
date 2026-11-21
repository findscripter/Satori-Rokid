# 六分钟实验量化分析

对象为 2026-11-20 一段 Rokid 录像的前 360 秒。一次 dialogue_request 计一次接纳请求。视频中的六条完成答复属于同一个人执行同一任务的连续回合。

## 运行与内容

- 7 次请求，6 次产生答复（85.7%），包括 5 次直接回答和 1 次澄清。另一次在 8.82 秒后因任务或指令变化撤销。
- 6 条答复各附 1 张当轮图，且各有匹配的开始和结束播报回调。本次检索判断均返回无需外部搜索。
- AI 辅助初审按轮判定：把回答中主要、肯定、可由单张图验证的现场陈述与当轮输入原图核对，标为支持、矛盾或不可评。“看不清”、提问及操作建议本身不计视觉命中。
- 5 轮给出了有图像依据的主要现场陈述，0 轮出现可见的明确反例，1 轮不可评。可评分覆盖率为 5/6（83.3%）；可评的主要陈述中，5/5 与图像相符。此值只适用于这段记录。
- 整体操作指导正确率、任务成功率与用户实际说完话到听见声音的延迟均缺少独立真值。若要评操作正确率，可由两位评审根据对应机型手册和连续录像，独立标注当步状态与可接受动作，再对照助手的每条明确指令。

## 设备观测到的耗时（秒）

|区间|中位数|平均值|最短–最长|
|---|---:|---:|---:|
|请求 → 取得取景记录|0.53|0.56|0.37–0.81|
|检索判断调用|4.36|5.56|3.01–11.47|
|最终问答调用|8.05|7.98|4.71–10.33|
|请求 → 文字答复|13.27|14.13|9.40–22.50|
|请求 → 开始播报|13.46|14.32|9.50–22.74|
|文字答复 → 开始播报|0.19|0.19|0.10–0.24|
|当轮图片采样 → 开始播报|13.44|14.07|9.00–22.30|
|播报开始 → 结束回调|20.20|19.52|14.91–22.63|

六轮累计等待文字 84.80 秒。检索判断占 33.37 秒，最终问答占 47.89 秒，合计 95.8%。两个调用阶段的时长含网络和处理，无法据此分离纯推理耗时。

用同轮请求 ID 配对 dialogue_request、dialogue_visual_snapshot、web_search_plan_request → web_search_plan、dialogue_route → dialogue_response。图片按 frames.jsonl.ts_ns。播报按答复文字和播报 ID 配对 audio_started 和 audio_done，排除另一次独立设备提示。所有时差使用设备单调时钟；转写发生于请求之前。各阶段中位数不可相加作为总中位数。

## 逐轮复核

|轮次|类型|请求至文字（秒）|请求至开播（秒）|主要现场陈述核对|
|---|---|---:|---:|---|
|d2|撤销|—|—|无最终答复。录音或任务指令改变后撤销。 [原图](https://findscripter.github.io/Satori-Rokid/replay/session/f2-522944043.jpg)|
|d3|answer|9.40|9.50|支持。正面放射状网罩可见。无法根据该图判定卡扣或螺丝类型。 [原图](https://findscripter.github.io/Satori-Rokid/replay/session/f4-522963574.jpg)|
|d4|answer|22.50|22.75|支持。网罩基本正对相机，中心圆盘清晰可见。 [原图](https://findscripter.github.io/Satori-Rokid/replay/session/f6-523006834.jpg)|
|d5|answer|11.88|12.09|支持。可见外圈的侧面轮廓及后方电机壳。 [原图](https://findscripter.github.io/Satori-Rokid/replay/session/f7-523081490.jpg)|
|d6|answer|12.78|12.98|支持。用户双手托住与机身分离的前网罩、扇叶外露。中心白色件的具体用途另无真值。 [原图](https://findscripter.github.io/Satori-Rokid/replay/session/f8-523123751.jpg)|
|d7|clarify|14.48|14.66|不可评。未对阻碍原因给出可评分的肯定判断。画面已经显示前罩落地、中心有小段突起；手中物身份和阻力来源无法验证。 [原图](https://findscripter.github.io/Satori-Rokid/replay/session/f10-523189809.jpg)|
|d8|answer|13.76|13.94|支持。前罩在地面，扇叶在机身上，轴心可见小段反光突起。材料和固定件类型无法据图验证。 [原图](https://findscripter.github.io/Satori-Rokid/replay/session/f11-523241006.jpg)|

[逐轮指标与原始事件行号](https://findscripter.github.io/Satori-Rokid/replay/downloads/per-turn-metrics.csv) · [下载原始事件，见包内 session/events.jsonl](https://github.com/findscripter/Satori-Rokid/releases/download/replay-2026-11-20/video-and-synced-data.zip) · [公开回放](https://findscripter.github.io/Satori-Rokid/replay/)
