using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace SatoriReplay
{
    public sealed class ReplayHud:MonoBehaviour
    {
        public TextAsset demoReport;
        public string reportPathOverride;
        private ReplayPlayback playback;
        private string reportPath;
        private Font font;
        private Text provenance,sourceInfo,session,task,scene,action,step,confidence,confirmed,observed,decision,evidence,imageNote,playLabel,speedLabel;
        private Slider timeline;
        private RawImage picture;
        private Texture2D loadedImage;
        private int lastFrame=-1,speedIndex=1;
        private readonly float[] speeds={0.5f,1,2,4};
        private readonly Color ink=new Color(0.88f,0.93f,0.94f),muted=new Color(0.55f,0.68f,0.72f),mint=new Color(0.45f,0.91f,0.71f);
        public int CurrentFrame=>playback?.Index??-1;
        public bool IsPlaying=>playback!=null&&playback.Playing;
        public ReplayReport LoadedReport=>playback?.Report;
        public ReplayRecord CurrentRecord=>playback?.Current;
        public string LoadedReportPath=>reportPath;
        public Texture2D CurrentImage=>loadedImage;
        public bool HasLoadedImage=>loadedImage!=null&&picture!=null&&picture.enabled&&picture.texture==loadedImage;
        public static string Argument(string key){var a=Environment.GetCommandLineArgs();for(int i=0;i+1<a.Length;i++)if(a[i]==key)return a[i+1];return null;}
        private void Start()
        {
            Application.runInBackground=true;
            font=Font.CreateDynamicFontFromOSFont(new[]{"Microsoft YaHei","Microsoft YaHei UI","SimHei","Arial"},24);
            BuildUI();ReloadReport();
            var capture=Argument("-capturePath");if(!string.IsNullOrWhiteSpace(capture))StartCoroutine(Capture(capture));
        }
        private RectTransform Box(string name,Transform parent,float x,float y,float w,float h,Color? color=null)
        {
            var go=new GameObject(name,typeof(RectTransform));go.transform.SetParent(parent,false);
            var rect=go.GetComponent<RectTransform>();rect.anchorMin=rect.anchorMax=new Vector2(0,1);rect.pivot=new Vector2(0,1);rect.anchoredPosition=new Vector2(x,-y);rect.sizeDelta=new Vector2(w,h);
            if(color.HasValue){var image=go.AddComponent<Image>();image.color=color.Value;image.raycastTarget=false;}
            return rect;
        }
        private Text Label(Transform parent,string name,string text,float x,float y,float w,float h,int size=22,Color? color=null)
        {
            var t=Box(name,parent,x,y,w,h).gameObject.AddComponent<Text>();t.font=font;t.fontSize=size;t.text=text;t.color=color??ink;t.supportRichText=false;t.horizontalOverflow=HorizontalWrapMode.Wrap;t.verticalOverflow=VerticalWrapMode.Truncate;t.raycastTarget=false;return t;
        }
        private Text Button(Transform parent,string caption,float x,float y,float w,UnityEngine.Events.UnityAction callback)
        {
            var rect=Box(caption,parent,x,y,w,46,new Color(0.13f,0.22f,0.25f));rect.GetComponent<Image>().raycastTarget=true;
            var control=rect.gameObject.AddComponent<Button>();control.onClick.AddListener(callback);
            var label=Label(rect,"Label",caption,0,0,w,46,20);label.alignment=TextAnchor.MiddleCenter;return label;
        }
        private void BuildUI()
        {
            var canvasObject=new GameObject("Satori Replay Canvas",typeof(Canvas),typeof(CanvasScaler),typeof(GraphicRaycaster));canvasObject.transform.SetParent(transform,false);
            canvasObject.GetComponent<Canvas>().renderMode=RenderMode.ScreenSpaceOverlay;
            var scaler=canvasObject.GetComponent<CanvasScaler>();scaler.uiScaleMode=CanvasScaler.ScaleMode.ScaleWithScreenSize;scaler.referenceResolution=new Vector2(1440,940);scaler.screenMatchMode=CanvasScaler.ScreenMatchMode.Expand;
            var root=Box("Fixed HUD",canvasObject.transform,0,0,1440,940,new Color(0.035f,0.055f,0.065f));root.anchorMin=root.anchorMax=root.pivot=new Vector2(0.5f,0.5f);root.anchoredPosition=Vector2.zero;
            if(FindObjectOfType<EventSystem>()==null)new GameObject("Replay Input",typeof(EventSystem),typeof(StandaloneInputModule));
            Label(root,"Title","SATORI  /  环境与任务回放",32,22,950,52,32);
            Label(root,"Mode","离线研究客户端 · 未连接实时相机与麦克风 · 按记录回放",32,78,1050,32,18,muted);
            provenance=Label(root,"Provenance","等待加载",1010,34,398,40,20,mint);provenance.alignment=TextAnchor.UpperRight;
            var panelColor=new Color(0.075f,0.105f,0.12f);
            var left=Box("Evidence Panel",root,32,130,792,480,panelColor);
            Label(left,"Heading","输入图像证据",24,16,720,28,20,mint);
            sourceInfo=Label(left,"Source","",24,48,744,24,16,muted);
            Box("Image Backdrop",left,24,74,744,360,new Color(0.02f,0.035f,0.045f));
            var imageArea=Box("Image",left,24,74,744,360);picture=imageArea.gameObject.AddComponent<RawImage>();picture.raycastTarget=false;picture.color=Color.white;
            imageNote=Label(left,"Image Note","该记录没有关联图像\n\n只展示本地报告与关联图片。\n状态回放不代表实时视觉识别。",70,165,650,150,23,muted);
            session=Label(left,"Frame","",24,442,744,30,18,muted);
            var right=Box("Task State",root,848,130,560,480,panelColor);
            Label(right,"Heading","当前任务状态",24,20,500,32,20,mint);
            task=Label(right,"Task","",24,66,512,68,26);
            Label(right,"Scene Label","场景",24,141,500,26,17,muted);scene=Label(right,"Scene","",24,171,512,58,21);
            Label(right,"Action Label","用户正在做",24,236,500,26,17,muted);action=Label(right,"Action","",24,266,512,58,21);
            step=Label(right,"Step","",24,333,512,70,21);
            confidence=Label(right,"Confidence","",24,418,512,50,18,muted);
            var states=Box("Completion",root,32,630,792,152,panelColor);
            Label(states,"Confirmed Label","用户已确认完成",24,18,340,28,19,mint);confirmed=Label(states,"Confirmed","",24,62,340,70,23);
            Label(states,"Observed Label","模型观察完成 · 尚待确认",400,18,368,28,19,mint);observed=Label(states,"Observed","",400,62,368,70,23);
            var assist=Box("Assistance",root,848,630,560,152,panelColor);
            decision=Label(assist,"Decision","",24,18,512,94,21);evidence=Label(assist,"References","",24,119,512,26,17,muted);
            Label(root,"Timeline Label","时间轴 · 拖动定位 / 按记录播放",32,802,1100,28,17,muted);
            var track=Box("Timeline",root,32,844,1376,15,new Color(0.15f,0.25f,0.28f));
            track.GetComponent<Image>().raycastTarget=true;
            timeline=track.gameObject.AddComponent<Slider>();timeline.minValue=0;timeline.maxValue=1;timeline.wholeNumbers=true;
            var fill=Box("Fill",track,0,0,0,0,mint);
            fill.pivot=new Vector2(0.5f,0.5f);fill.anchorMin=Vector2.zero;fill.anchorMax=Vector2.one;
            fill.sizeDelta=Vector2.zero;fill.anchoredPosition=Vector2.zero;
            timeline.fillRect=fill;
            var handleArea=Box("Handle Slide Area",track,0,0,0,12);
            handleArea.pivot=new Vector2(0.5f,0.5f);handleArea.anchorMin=Vector2.zero;handleArea.anchorMax=Vector2.one;
            handleArea.sizeDelta=new Vector2(0,12);handleArea.anchoredPosition=Vector2.zero;
            var handle=Box("Handle",handleArea,0,0,24,0,Color.white);
            handle.pivot=new Vector2(0.5f,0.5f);handle.sizeDelta=new Vector2(24,0);handle.anchoredPosition=Vector2.zero;
            handle.GetComponent<Image>().raycastTarget=true;timeline.handleRect=handle;timeline.targetGraphic=handle.GetComponent<Image>();timeline.direction=Slider.Direction.LeftToRight;
            timeline.onValueChanged.AddListener(v=>{if(playback!=null&&Mathf.RoundToInt(v)!=playback.Index){playback.Seek(Mathf.RoundToInt(v));Render();}});
            Button(root,"上一条",32,879,140,PreviousFrame);playLabel=Button(root,"播放 / 重播",184,879,180,TogglePlayback);Button(root,"下一条",376,879,140,NextFrame);
            speedLabel=Button(root,"速度 1×",532,879,170,()=>{speedIndex=(speedIndex+1)%speeds.Length;if(playback!=null)playback.Speed=speeds[speedIndex];speedLabel.text="速度 "+speeds[speedIndex]+"×";});
            Button(root,"重新读取",718,879,150,ReloadReport);Label(root,"Keyboard","空格 播放/暂停  ·  ← → 逐条  ·  Home 开始",900,891,508,30,17,muted);
            // Keep longer model output readable within the existing evidence panels.
            foreach(var label in new[]{task,scene,action,step,decision,evidence,confirmed,observed})
            {label.resizeTextForBestFit=true;label.resizeTextMinSize=16;label.resizeTextMaxSize=label.fontSize;}
        }
        public void ReloadReport()
        {
            try
            {
                reportPath=Argument("-reportPath");
                if(string.IsNullOrWhiteSpace(reportPath))reportPath=reportPathOverride;
                ReplayReport report;
                if(!string.IsNullOrWhiteSpace(reportPath)){reportPath=Path.GetFullPath(reportPath);report=ReportParser.Load(reportPath);}
                else if(File.Exists(Path.Combine(Application.streamingAssetsPath,"demo-report.json"))){reportPath=Path.Combine(Application.streamingAssetsPath,"demo-report.json");report=ReportParser.Load(reportPath);}
                else if(demoReport!=null){reportPath=Path.Combine(Application.dataPath,"Satori/Data/demo-report.json");report=ReportParser.Parse(demoReport.text);}
                else throw new ArgumentException("没有报告：请提供 -reportPath，或绑定 demo-report.json。");
                ApplyReport(report,reportPath);
            }
            catch(Exception e){ClearReport(e);}
        }
        private void ApplyReport(ReplayReport report,string path)
        {
            reportPath=path;playback=new ReplayPlayback(report);playback.Speed=speeds[speedIndex];lastFrame=-1;
            if(int.TryParse(Argument("-startFrame"),out var start))playback.Seek(start);SetPlaybackControls(true);Render();
        }
        public void LoadReport(string path)
        {
            try
            {
                if(string.IsNullOrWhiteSpace(path))throw new ArgumentException("报告路径不能为空。",nameof(path));
                reportPathOverride=path;
                var absolute=Path.GetFullPath(path);
                reportPathOverride=absolute;
                var report=ReportParser.Load(absolute);
                if(provenance!=null)ApplyReport(report,absolute);
            }
            catch(Exception e){ClearReport(e);throw;}
        }
        private void SetPlaybackControls(bool enabled)
        {
            if(timeline!=null)timeline.interactable=enabled;
            foreach(var button in GetComponentsInChildren<Button>(true))
                if(button.name!="重新读取")button.interactable=enabled;
        }
        private void ClearReport(Exception error)
        {
            playback=null;reportPath=null;lastFrame=-1;
            ReleaseImage();
            foreach(var label in new[]{sourceInfo,session,task,scene,action,step,confidence,confirmed,observed,decision,evidence})
                if(label!=null)label.text="暂无记录";
            if(provenance!=null)provenance.text="报告未加载";
            if(imageNote!=null){imageNote.text=error.Message;imageNote.gameObject.SetActive(true);}
            if(timeline!=null){timeline.minValue=0;timeline.maxValue=1;timeline.SetValueWithoutNotify(0);}
            if(playLabel!=null)playLabel.text="播放 / 重播";
            SetPlaybackControls(false);
            Debug.LogWarning("Satori replay: "+error.Message);
        }
        public void SeekFrame(int index){if(playback==null)return;playback.Seek(index);Render();}
        public void PreviousFrame(){if(playback==null)return;playback.Previous();Render();}
        public void NextFrame(){if(playback==null)return;playback.Next();Render();}
        public void TogglePlayback(){if(playback==null)return;playback.Toggle();Render();}
        private void Update()
        {
            if(playback==null)return;
            if(Input.GetKeyDown(KeyCode.Space))TogglePlayback();if(Input.GetKeyDown(KeyCode.LeftArrow))PreviousFrame();if(Input.GetKeyDown(KeyCode.RightArrow))NextFrame();if(Input.GetKeyDown(KeyCode.Home)){playback.Seek(0);Render();}
            playback.Advance(Time.unscaledDeltaTime);if(lastFrame!=playback.Index)Render();
        }
        private static string V(string text,string fallback="暂无记录")=>string.IsNullOrWhiteSpace(text)?fallback:text;
        private static string Join(string[] values)=>values.Length==0?"暂无":string.Join(" · ",values);
        private static string Kind(string kind)=>kind=="guide"?"提供指导":kind=="clarify"?"请求澄清":kind=="complete"?"任务完成":"继续观察";
        private void Render()
        {
            var r=playback.Current;provenance.text=playback.Report.synthetic?"合成数据 · 仅验证管线":"非合成数据 · 离线回放";
            var sourceKind=(playback.Report.source_kind??"").ToLowerInvariant();
            if(playback.Report.synthetic)sourceInfo.text="合成管线样例，不是模型效果评估。";
            else if(sourceKind.Contains("public")||sourceKind.Contains("dataset"))sourceInfo.text="公开数据集/视频 · 非用户实拍  /  "+V(playback.Report.source_label,"来源未详注");
            else sourceInfo.text=V(playback.Report.source_label,"数据来源未注明，不视作用户实拍。");
            task.text=V(r.task_title);scene.text=V(r.scene);action.text=V(r.action);step.text="当前步骤  "+V(r.current_step_id,"尚未确定")+"\n"+V(r.current_step_instruction);
            confidence.text="模型信心 "+r.confidence.ToString("P0")+"  /  "+(r.uncertain?"存在不确定性":"未标记不确定")+"\n信心值不是经校准的正确概率";
            confirmed.text=Join(r.confirmed_completed_step_ids);var pending=new List<string>();var done=new HashSet<string>(r.confirmed_completed_step_ids);foreach(var id in r.observed_completed_step_ids)if(!done.Contains(id))pending.Add(id);observed.text=Join(pending.ToArray());
            decision.text="辅助决策 / "+Kind(r.decision_kind)+"\n"+V(r.decision_message,"保持观察，当前没有新提示。");evidence.text="证据："+Join(r.evidence_frame_ids);
            session.text=r.frame_id+"  /  "+TimeSpan.FromMilliseconds(r.timestamp_ms).ToString(@"mm\:ss\.fff")+"  /  "+(playback.Index+1)+" / "+playback.Report.records.Length;
            timeline.maxValue=Mathf.Max(1,playback.Report.records.Length-1);timeline.SetValueWithoutNotify(playback.Index);playLabel.text=playback.Playing?"暂停":"播放 / 重播";
            if(lastFrame!=playback.Index){lastFrame=playback.Index;LoadPicture(r.image_path);}
        }
        private void LoadPicture(string path)
        {
            ReleaseImage();imageNote.gameObject.SetActive(true);
            imageNote.text="该记录没有关联图像\n\n只展示本地报告与关联图片。\n状态回放不代表实时视觉识别。";
            if(string.IsNullOrWhiteSpace(path))return;
            try
            {
                var full=Path.IsPathRooted(path)?path:Path.Combine(Path.GetDirectoryName(reportPath),path);
                if(!File.Exists(full))throw new IOException("图片不存在");if(new FileInfo(full).Length>20*1024*1024)throw new IOException("图片超过20MB");
                loadedImage=new Texture2D(2,2,TextureFormat.RGB24,false);if(!loadedImage.LoadImage(File.ReadAllBytes(full)))throw new IOException("图片无法解码");
                picture.texture=loadedImage;picture.enabled=true;imageNote.gameObject.SetActive(false);
                var fit=picture.GetComponent<AspectRatioFitter>();if(fit!=null)fit.enabled=false;
                // A disabled AspectRatioFitter can leave stretch anchors behind. Reset the
                // actual RectTransform before applying pixel-sized, aspect-preserving bounds.
                var imageRect=picture.rectTransform;
                imageRect.anchorMin=imageRect.anchorMax=new Vector2(0,1);
                imageRect.pivot=new Vector2(0,1);
                float ratio=Mathf.Min(744f/loadedImage.width,360f/loadedImage.height);
                var fittedSize=new Vector2(loadedImage.width*ratio,loadedImage.height*ratio);
                imageRect.sizeDelta=fittedSize;
                imageRect.anchoredPosition=new Vector2(24+(744-fittedSize.x)/2,-(74+(360-fittedSize.y)/2));
            }
            catch(Exception){imageNote.text="关联图片无法读取，状态记录仍可回放。";ReleaseImage();}
        }
        private void ReleaseImage()
        {
            if(picture!=null){picture.texture=null;picture.enabled=false;}
            if(loadedImage!=null)Destroy(loadedImage);
            loadedImage=null;
        }
        private IEnumerator Capture(string path)
        {
            bool captureOnly=Array.IndexOf(Environment.GetCommandLineArgs(),"-captureOnly")>=0;
            bool success=false,reportLoaded=false,hasChineseGlyphs=false,splashFinished=false;
            int width=0,height=0,attempts=0,brightSamples=0,pixelRange=0;
            string failure="",absolutePath=path;
            float started=Time.realtimeSinceStartup;

            // A player can reach Start while its splash screen still owns the backbuffer.
            while(!Application.isEditor&&!UnityEngine.Rendering.SplashScreen.isFinished&&Time.realtimeSinceStartup-started<30f)
                yield return null;
            splashFinished=Application.isEditor||UnityEngine.Rendering.SplashScreen.isFinished;
            if(!splashFinished)failure="Splash screen did not finish within 30 seconds.";

            if(splashFinished)
            {
                // Let the post-splash render target and dynamic font atlas settle.
                float settleUntil=Time.realtimeSinceStartup+1f;
                while(Time.realtimeSinceStartup<settleUntil){Canvas.ForceUpdateCanvases();yield return null;}
                for(int i=0;i<3;i++){Canvas.ForceUpdateCanvases();yield return new WaitForEndOfFrame();}
                try
                {
                    absolutePath=Path.GetFullPath(path);
                    Directory.CreateDirectory(Path.GetDirectoryName(absolutePath));
                    reportLoaded=playback!=null&&CurrentFrame>=0;
                    const string chinese="环境任务回放合成数据用户确认观察辅助决策证据";
                    if(font!=null)
                    {
                        font.RequestCharactersInTexture(chinese,24,FontStyle.Normal);
                        hasChineseGlyphs=true;
                        foreach(char c in chinese)if(!font.HasCharacter(c)){hasChineseGlyphs=false;break;}
                    }
                    if(!reportLoaded)failure="Report was not loaded.";
                    else if(!hasChineseGlyphs)failure="Selected font lacks required Chinese glyphs.";
                }
                catch(Exception e){failure="Capture preparation failed: "+e.Message;}

                // Retry only an empty/black render target; do not call it a successful screenshot.
                while(failure==""&&attempts<8&&!success)
                {
                    Canvas.ForceUpdateCanvases();
                    yield return new WaitForEndOfFrame();
                    Texture2D texture=null;
                    attempts++;
                    try
                    {
                        texture=ScreenCapture.CaptureScreenshotAsTexture();
                        if(texture!=null)
                        {
                            width=texture.width;height=texture.height;
                            var pixels=texture.GetPixels32();
                            int stride=Mathf.Max(1,pixels.Length/65536),lowest=255,highest=0;
                            brightSamples=0;
                            for(int i=0;i<pixels.Length;i+=stride)
                            {
                                var p=pixels[i];int level=Mathf.Max(p.r,Mathf.Max(p.g,p.b));
                                lowest=Mathf.Min(lowest,level);highest=Mathf.Max(highest,level);
                                if(level>64)brightSamples++;
                            }
                            pixelRange=highest-lowest;
                            if(width>0&&height>0&&brightSamples>=32&&pixelRange>=32)
                            {
                                File.WriteAllBytes(absolutePath,texture.EncodeToPNG());success=true;
                            }
                        }
                    }
                    catch(Exception e){failure="ScreenCapture failed: "+e.Message;}
                    finally{if(texture!=null)Destroy(texture);}
                    if(!success&&failure=="")yield return new WaitForSecondsRealtime(0.25f);
                }
                if(!success&&failure=="")failure="Render target remained black, uniform, or empty after 8 attempts.";
            }

            string Quote(string value)
            {
                using(var buffer=new MemoryStream())
                {
                    new System.Runtime.Serialization.Json.DataContractJsonSerializer(typeof(string)).WriteObject(buffer,value??"");
                    return System.Text.Encoding.UTF8.GetString(buffer.ToArray());
                }
            }
            try
            {
                absolutePath=Path.GetFullPath(path);
                Directory.CreateDirectory(Path.GetDirectoryName(absolutePath));
                string metadata="{\n  \"success\":"+success.ToString().ToLowerInvariant()+
                    ",\n  \"failure\":"+Quote(failure)+
                    ",\n  \"CurrentFrame\":"+CurrentFrame+
                    ",\n  \"ReportLoaded\":"+reportLoaded.ToString().ToLowerInvariant()+
                    ",\n  \"synthetic\":"+(playback!=null&&playback.Report.synthetic).ToString().ToLowerInvariant()+
                    ",\n  \"font_name\":"+Quote(font!=null?font.name:"")+
                    ",\n  \"font_names\":"+Quote(font!=null?string.Join(", ",font.fontNames):"")+
                    ",\n  \"chinese_glyphs_available\":"+hasChineseGlyphs.ToString().ToLowerInvariant()+
                    ",\n  \"splash_finished\":"+splashFinished.ToString().ToLowerInvariant()+
                    ",\n  \"image_width\":"+width+",\n  \"image_height\":"+height+
                    ",\n  \"bright_samples\":"+brightSamples+",\n  \"pixel_range\":"+pixelRange+
                    ",\n  \"capture_attempts\":"+attempts+
                    ",\n  \"unity_version\":"+Quote(Application.unityVersion)+"\n}";
                File.WriteAllText(absolutePath+".json",metadata);
            }
            catch(Exception e){success=false;failure="Capture metadata could not be saved: "+e.Message;}
            if(success)Debug.Log("SATORI_CAPTURE_PASS "+absolutePath);
            else Debug.LogError("SATORI_CAPTURE_FAILED "+failure);
            if(captureOnly)Application.Quit(success?0:1);
        }
        private void OnDestroy(){ReleaseImage();if(font!=null)Destroy(font);}
    }
}
