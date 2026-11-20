# Satori 眼镜端助手

打开眼镜上的 Satori 后可以直接说当前需求。通用模式接收普通问句、陈述和连续追问，优先使用提问时的第一视角图像，回答后自动朗读。录像、停止录像等设备控制在本机执行。联网检索由当轮请求决定。

## 构建与安装

工程采用 JDK 17、Gradle 8.4、Android SDK Platform 34 和 Build Tools 34.0.0，目标为 arm64-v8a，最低 Android API 31。以下命令在仓库根目录的 PowerShell 中执行。先设置 `ANDROID_HOME`，将 Gradle 和 Android Platform Tools 加入 `PATH`。

```powershell
gradle -p Glasses/SatoriCapture :app:testDebugUnitTest :app:assembleDebug
adb devices -l
$serial = Read-Host "设备序列号"
adb -s $serial install -r -g Glasses/SatoriCapture/app/build/outputs/apk/debug/app-debug.apk
```

已有安装升级时，使用相同签名，并通过 `-PsatoriVersionCode=<版本号>` 和 `-PsatoriVersionName=<版本名>` 构建。版本号应高于已有安装。配置和导出工具使用 Android 的 `run-as`，以下步骤适用于 debug APK。

## 首次配置

眼镜需连接可上网的 Wi-Fi。准备一个能接收图片的 HTTPS Chat Completions 服务和该服务提供的模型 ID。密钥通过本机 Windows 用户的 DPAPI 加密存储。用 Python 3 在仓库根目录执行以下命令，在隐藏输入提示中输入密钥。

```powershell
python -c "import sys; from getpass import getpass; from pathlib import Path; sys.path.insert(0,'Tools'); import satori_credentials as c; c.save(getpass('Provider API key: '), Path('UserSettings/SatoriStandalone/provider.dpapi'))"
```

在 `UserSettings/SatoriStandalone/provider.json` 填入自己的服务地址和模型 ID。以下三个服务参数均为示意值。

```json
{
  "base_url": "https://your-provider.example/v1",
  "key_store": "UserSettings/SatoriStandalone/provider.dpapi",
  "planner": "your-vision-model",
  "monitor": "your-vision-model"
}
```

运行以下命令将配置导入眼镜，启用通用对话和网页检索。工具会重启 App，宜在开始一轮录像之前执行。每次应用配置使用一个新目录保存验证结果。

```powershell
python Tools/standalone_glasses.py --serial $serial provision
python Tools/standalone_glasses.py --serial $serial status
$configRun = "UserSettings/SatoriStandalone/config-" + (Get-Date -Format yyyyMMdd-HHmmss)
python Tools/configure_classroom_demo.py --serial $serial --manual-source web --mode guided --apply --evidence $configRun
```

首次使用时确认相机和麦克风权限。之后可以拔掉 USB，保持眼镜联网并直接说话。通过状态输出可核对设备是否完成加密配置导入；网页模式的配置证据包含本次会话的核验结果。

## 实验数据与录像

说“开始录像”开始第一视角有声录像，说“停止录像”并等到保存提示。Satori 在独立会话中记录交互事件、状态、实际采集的图片和 IMU 数据。实验结束并确认录像已保存后，接回 USB，停下仍在写日志的 App，导出后再打开。

```powershell
$exportDir = "UserSettings/SatoriStandalone/export-" + (Get-Date -Format yyyyMMdd-HHmmss)
adb -s $serial shell am force-stop com.satori.glasses.capture
python Tools/standalone_glasses.py --serial $serial export --output $exportDir
python Tools/standalone_glasses.py --serial $serial list-videos
python Tools/standalone_glasses.py --serial $serial export-videos --output "$exportDir/videos"
python Tools/standalone_glasses.py --serial $serial launch
```

中文离线识别使用 Vosk。随包模型的来源、许可证和各文件校验值见 [模型清单](app/src/main/assets/vosk-model-small-cn/source-manifest.json)。
