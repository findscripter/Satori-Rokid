# Satori 离线回放客户端

当前工程使用已打开的 Unity 2022.3.62f3c1。回放界面使用项目现成 UGUI，运行时创建固定 HUD，读取本地报告和图片。另有可选 Mage 环境问答页；Mage 当前已关闭，默认进入任务回放。相机、麦克风和 Rokid 实机采集尚未接入。

## 菜单

- `Satori/Create Replay Scene`：创建并打开 `Assets/Satori/Scenes/SatoriReplay.unity`。如果 `Assets/Satori/Data/demo-report.json` 存在，会绑定为场景内置数据。应先保存当前场景再执行此菜单。
- `Satori/Run Core Checks`：22 项数据与播放边界检查，最后 3 项验证后端九帧合成报告。结果保存到 `Assets/Satori/Evidence/core-tests.json`。
- `Satori/Run Play Mode Checks`：Play Mode 中验证 UI 实例、合成标识、前后按钮、证据切换、播放/暂停和中文字体字形。结果保存到 `Assets/Satori/Evidence/play-tests.json`。
- `Satori/Build Windows Replay`：将回放场景构建至 `Builds/Windows/SatoriReplay.exe`，不修改项目包或 PlayerSettings。结果保存到 `Assets/Satori/Evidence/build-result.json`。
- `Satori/Run Default Replay Entry Checks`：在非Play状态检查保存的场景默认进入回放，Mage页保持关闭。
- `Satori/Run Report Load Failure Checks`：在Play中用临时本地夹具检查报告缺失、损坏、直接加载失败后的旧状态清理，以及随后成功加载的恢复。结果写入 `Evidence/code-fixes-20261017`，不调用模型。

## 报告与控制

加载优先级：命令行 `-reportPath <绝对路径>` → Inspector的 `reportPathOverride` → `StreamingAssets/demo-report.json` → 场景绑定的 TextAsset。数据格式遵循后端导出的 `{session_id, synthetic, records:[...]}`。路径相对报告目录解析。复制报告到其他目录时，需要同时调整或复制关联图片。

空格播放/暂停，方向键逐条，Home 回到开头。支持 0.5 / 1 / 2 / 4 倍速和时间轴定位。UI 分开显示用户确认和尚未确认的模型观察，重叠项只在确认列显示。synthetic=false 只表示输入文件的数据声明，界面仍为离线回放。

报告加载或重读失败时，清除旧文字、图片、来源路径和时间轴，禁用播放控件，保留“重新读取”；错误原因可见。重新加载有效报告后恢复控件。回放操作仍不向后端提交任务确认事件，当前不能作为完整任务交互客户端。

可选 `-startFrame <从0开始的序号>`；`-capturePath <PNG绝对路径>` 会在界面绘制后保存截图，配合 `-captureOnly` 截图后退出。截图需实际图形上下文，不支持 nographics。

## 验证边界

此前同一核心实现使用 2022.3.60f1c1 自带编译器和 Mono 运行过 22 项检查；此次恢复的新工程必须通过上述菜单重新验证。只能以本工程实际生成的 Evidence 文件和 Unity Console 为准，不能把旧结果当成本轮 Play Mode/构建通过。字体优先使用 Windows Microsoft YaHei / Microsoft YaHei UI / SimHei，运行时检查字形与实际截图共同确认中文显示。

默认九帧数据是合成管线回归样例，不能作为模型识别准确率、Satori 论文复现性能或眼镜实时能力证据。
