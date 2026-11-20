# 第三方代码与数据说明

## Satori参考任务计划

- 上游：https://github.com/VIDA-NYU/satori-assistance
- 位置：`Research/Reference/satori-assistance/pipelines/task/task_plans/`
- 许可：MIT，原版权说明与完整许可保留于 `Research/Reference/satori-assistance/LICENSE`。
- 用途：任务计划导入与对应测试；未包含上游完整研究系统。

## MCP for Unity

- 上游：https://github.com/CoplayDev/unity-mcp
- 位置：`Packages/com.coplaydev.unity-mcp/`
- 嵌入版本：`9.6.7-beta.4`，以包内 `package.json` 为准。
- 许可：MIT，完整许可位于包内 `LICENSE`。许可文本取自上游Git blob `e7f878d1da6b4d9488058157ce6b48cb584460b1`。

## 数据与模型

眼镜应用随附 Vosk 中文离线语音识别模型 `vosk-model-small-cn-0.22`，便于在设备上识别中文。原始下载地址、许可证和文件校验值见[模型清单](Glasses/SatoriCapture/app/src/main/assets/vosk-model-small-cn/source-manifest.json)及 [Apache-2.0 许可证](Glasses/SatoriCapture/app/src/main/assets/vosk-model-small-cn/LICENSE-APACHE-2.0.txt)。

公开数据评测所需的其他数据及模型通过各自来源获取。仓库内的合成演示和语音测试样例用于功能验证。网页回放中的六分钟视频、音轨、转写和取景帧来自本项目的实机实验。

README中的Unity运行截图含EgoDex少量视频画面，用于展示研究界面；作者、来源及结果说明见 [图片说明](docs/images/README.md)。

以上第三方许可仅适用于各自文件，不构成对本仓库所有自有代码的统一重新许可。
