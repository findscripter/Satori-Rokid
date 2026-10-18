using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using Debug=UnityEngine.Debug;

namespace SatoriReplay
{
    public sealed class MageContextHud:MonoBehaviour
    {
        public string sessionPath="Research/Data/real-task/clean_surface_0_dense_2fps/session.json";
        public long atMs=9000;
        public int window=8;
        public string pythonExecutableOverride;
        public float helperTimeoutSeconds=300;
        public string initialQuestion="请描述这段视频中的环境、可见物体和人的动作。区分能确定与不能确定的信息，并引用输入帧作为证据。";

        public string RequestState{get;private set;}="idle";
        public MageResponse LastResult{get;private set;}
        public string RequestId{get;private set;}
        public string RequestFile{get;private set;}
        public string ResponseFile{get;private set;}
        public string LastResponseJson{get;private set;}
        public MageRequest LastRequest{get;private set;}
        public int? HelperExitCode{get;private set;}
        public bool ResponseAcceptedFromOwnProcess{get;private set;}
        public int CompletedRequestCount{get;private set;}
        public int SelectedFrame{get;private set;}=-1;
        public MageInputFrame[] InputFrames=>previewFrames;
        public Texture2D CurrentImage=>SelectedFrame>=0&&SelectedFrame<textures.Count?textures[SelectedFrame]:null;
        public string SelectedFrameId=>SelectedFrame>=0&&SelectedFrame<previewFrames.Length?previewFrames[SelectedFrame].frame_id:null;
        public string DisplayedRawAnswer=>answerText!=null?answerText.text:"";
        public string DisplayedDetails=>detailsText!=null?detailsText.text:"";
        public Button AskButton=>askButton;
        public string QuestionText{get=>questionInput!=null?questionInput.text:initialQuestion;set{initialQuestion=value;if(questionInput!=null)questionInput.text=value;}}
        public string ProjectRoot=>Path.GetFullPath(Path.Combine(Application.dataPath,".."));

        private readonly Dictionary<GameObject,bool> hiddenReplay=new Dictionary<GameObject,bool>();
        private readonly List<Texture2D> textures=new List<Texture2D>();
        private MageInputFrame[] previewFrames=Array.Empty<MageInputFrame>();
        private MageInputFrame[] requestedFrames=Array.Empty<MageInputFrame>();
        private Font font;
        private Text statusText,summaryText,detailsText,answerText,mainCaption,runText;
        private InputField questionInput;
        private Button askButton;
        private Text askLabel;
        private RawImage mainImage;
        private readonly List<RawImage> thumbnails=new List<RawImage>();
        private readonly List<Text> thumbnailLabels=new List<Text>();
        private RectTransform detailsContent,answerContent;
        private Process helper;
        private float helperStarted;
        private int stdoutCharacters,stderrCharacters;
        private bool initialized;
        private readonly Color ink=new Color(0.9f,0.94f,0.95f),muted=new Color(0.59f,0.70f,0.74f),mint=new Color(0.45f,0.91f,0.71f);

        private void Start()
        {
            if(!Application.isPlaying)return;
            Application.runInBackground=true;
            font=Font.CreateDynamicFontFromOSFont(new[]{"Microsoft YaHei","Microsoft YaHei UI","SimHei","Arial"},24);
            BuildUi();initialized=true;HideReplay();
            try{LoadPreview();SetIdle();}catch(Exception){Fail("preview_unavailable","无法读取本地素材窗口，请检查 session 路径。",null);}
        }
        private void OnEnable(){if(Application.isPlaying&&initialized)HideReplay();}
        private void HideReplay()
        {
            foreach(var replay in Resources.FindObjectsOfTypeAll<ReplayHud>())
            {
                if(replay.gameObject.scene!=gameObject.scene||hiddenReplay.ContainsKey(replay.gameObject))continue;
                hiddenReplay[replay.gameObject]=replay.gameObject.activeSelf;replay.gameObject.SetActive(false);
            }
        }
        private void OnDisable()
        {
            if(helper!=null){StopOwnHelper();if(RequestState=="loading")Fail("client_cancelled","已停止本地请求助手；服务器未被终止。",LastResult?.run_directory);}
            foreach(var pair in hiddenReplay)if(pair.Key!=null)pair.Key.SetActive(pair.Value);
            hiddenReplay.Clear();
        }
        private void OnDestroy(){foreach(var texture in textures)if(texture!=null)Destroy(texture);if(font!=null)Destroy(font);}
        public void ReturnToReplay()=>gameObject.SetActive(false);

        private RectTransform Box(string name,Transform parent,float x,float y,float width,float height,Color? color=null)
        {
            var go=new GameObject(name,typeof(RectTransform));go.transform.SetParent(parent,false);
            var rect=go.GetComponent<RectTransform>();rect.anchorMin=rect.anchorMax=rect.pivot=new Vector2(0,1);
            rect.anchoredPosition=new Vector2(x,-y);rect.sizeDelta=new Vector2(width,height);
            if(color.HasValue){var image=go.AddComponent<Image>();image.color=color.Value;image.raycastTarget=false;}
            return rect;
        }
        private Text Label(Transform parent,string name,string text,float x,float y,float width,float height,int size=20,Color? color=null)
        {
            var label=Box(name,parent,x,y,width,height).gameObject.AddComponent<Text>();label.font=font;label.fontSize=size;label.text=text;
            label.color=color??ink;label.supportRichText=false;label.raycastTarget=false;label.horizontalOverflow=HorizontalWrapMode.Wrap;label.verticalOverflow=VerticalWrapMode.Truncate;return label;
        }
        private Button MakeButton(Transform parent,string name,string text,float x,float y,float width,float height,UnityEngine.Events.UnityAction clicked,out Text label)
        {
            var rect=Box(name,parent,x,y,width,height,new Color(0.12f,0.24f,0.25f));var image=rect.GetComponent<Image>();image.raycastTarget=true;
            var button=rect.gameObject.AddComponent<Button>();button.targetGraphic=image;button.onClick.AddListener(clicked);
            var colors=button.colors;colors.disabledColor=new Color(0.32f,0.39f,0.4f);button.colors=colors;
            label=Label(rect,"Label",text,0,0,width,height,20);label.alignment=TextAnchor.MiddleCenter;return button;
        }
        private Text ScrollText(Transform parent,string name,float x,float y,float width,float height,int fontSize,out RectTransform content)
        {
            var viewport=Box(name,parent,x,y,width,height,new Color(0.055f,0.08f,0.09f));viewport.GetComponent<Image>().raycastTarget=true;viewport.gameObject.AddComponent<RectMask2D>();
            var scroll=viewport.gameObject.AddComponent<ScrollRect>();scroll.horizontal=false;scroll.vertical=true;scroll.movementType=ScrollRect.MovementType.Clamped;scroll.scrollSensitivity=24;scroll.viewport=viewport;
            content=Box("Content",viewport,12,10,width-24,height-20);scroll.content=content;
            var text=Label(content,"Text","",0,0,width-24,height-20,fontSize);return text;
        }
        private void SetScroll(Text text,RectTransform content,string value,float minimum)
        {
            text.text=value??"";float height=Mathf.Max(minimum,text.preferredHeight+16);
            text.rectTransform.sizeDelta=new Vector2(text.rectTransform.sizeDelta.x,height);content.sizeDelta=new Vector2(content.sizeDelta.x,height+10);
        }
        private void BuildUi()
        {
            var go=new GameObject("Mage Context Canvas",typeof(Canvas),typeof(CanvasScaler),typeof(GraphicRaycaster));go.transform.SetParent(transform,false);
            var canvas=go.GetComponent<Canvas>();canvas.renderMode=RenderMode.ScreenSpaceOverlay;canvas.sortingOrder=30;
            var scaler=go.GetComponent<CanvasScaler>();scaler.uiScaleMode=CanvasScaler.ScaleMode.ScaleWithScreenSize;scaler.referenceResolution=new Vector2(1440,940);scaler.screenMatchMode=CanvasScaler.ScreenMatchMode.Expand;
            var root=Box("Mage Fixed HUD",go.transform,0,0,1440,940,new Color(0.03f,0.05f,0.06f));root.anchorMin=root.anchorMax=root.pivot=new Vector2(0.5f,0.5f);root.anchoredPosition=Vector2.zero;
            if(FindObjectOfType<EventSystem>()==null)new GameObject("Mage Input",typeof(EventSystem),typeof(StandaloneInputModule));
            Label(root,"Title","Mage-VL  /  视觉上下文实验",24,20,950,48,31);
            var provenance=Label(root,"Provenance","公开数据集素材 · 非用户实拍",990,25,426,34,19,mint);provenance.alignment=TextAnchor.UpperRight;
            summaryText=Label(root,"Summary","模型 Mage-VL · 未发起请求 · 无 IMU 输入",24,75,1100,28,18,muted);
            Text unused;MakeButton(root,"Return To Replay","返回任务回放",1200,69,216,34,ReturnToReplay,out unused);
            var panel=new Color(0.07f,0.10f,0.115f);
            var left=Box("Visual Context",root,24,114,760,420,panel);mainCaption=Label(left,"Selected Frame","输入窗口预览",24,12,712,27,18,mint);
            Box("Main Backdrop",left,24,48,712,258,new Color(0.015f,0.025f,0.03f));mainImage=Box("Main Image",left,24,48,712,258).gameObject.AddComponent<RawImage>();mainImage.raycastTarget=false;
            for(int i=0;i<8;i++)
            {
                int selected=i;var thumb=Box("Frame "+i,left,24+i*89,322,82,84,new Color(0.11f,0.16f,0.18f));thumb.GetComponent<Image>().raycastTarget=true;
                var button=thumb.gameObject.AddComponent<Button>();button.onClick.AddListener(()=>SelectFrame(selected));
                var raw=Box("Thumbnail",thumb,3,3,76,43).gameObject.AddComponent<RawImage>();raw.raycastTarget=false;thumbnails.Add(raw);
                thumbnailLabels.Add(Label(thumb,"Timestamp","—",2,48,78,34,13,muted));
            }
            var right=Box("Context Answer",root,808,114,608,420,panel);Label(right,"Heading","环境、物体与动作",24,14,560,30,21,mint);
            detailsText=ScrollText(right,"Structured Details",16,52,576,350,21,out detailsContent);
            Label(root,"Question Label","问当前 8 帧窗口 · 不是任务完成判定",24,548,1100,26,18,muted);
            var input=Box("Question Input",root,24,582,1130,64,new Color(0.11f,0.17f,0.19f));input.GetComponent<Image>().raycastTarget=true;
            questionInput=input.gameObject.AddComponent<InputField>();questionInput.lineType=InputField.LineType.MultiLineNewline;questionInput.characterLimit=2000;
            questionInput.textComponent=Label(input,"Question Text","",14,8,1102,48,19);questionInput.text=initialQuestion;
            askButton=MakeButton(root,"Ask Mage-VL","询问 Mage-VL",1170,582,246,64,Ask,out askLabel);
            var answerPanel=Box("Raw Answer",root,24,666,1392,214,panel);Label(answerPanel,"Heading","本次原始回答 JSON · 可滚动查看完整内容",16,10,1360,29,18,mint);
            answerText=ScrollText(answerPanel,"Raw JSON Scroll",12,44,1368,158,17,out answerContent);
            statusText=Label(root,"Request Status","",24,893,1392,27,17,muted);runText=Label(root,"Run Reference","",24,923,1392,16,12,muted);
        }
        public static string FileSha256(string path)
        {
            using(var hash=SHA256.Create())using(var stream=File.OpenRead(path))return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-","").ToLowerInvariant();
        }
        private string SessionAbsolute()=>Path.GetFullPath(Path.IsPathRooted(sessionPath)?sessionPath:Path.Combine(ProjectRoot,sessionPath));
        private void LoadPreview()
        {
            if(window<1||window>8||atMs<0)throw new ArgumentException("Preview window must contain one to eight frames.");
            var path=SessionAbsolute();var session=MageJson.Read<MagePreviewSession>(MageJson.ReadFile(path));
            if(session?.frames==null)throw new ArgumentException("Session has no frames.");
            var candidates=new List<MageInputFrame>();long previous=-1;
            for(int i=0;i<session.frames.Length;i++)
            {
                var frame=session.frames[i];if(frame==null||frame.timestamp_ms<0||frame.timestamp_ms<=previous)throw new ArgumentException("Session timestamps must increase strictly.");previous=frame.timestamp_ms;
                if(frame.timestamp_ms>atMs)continue;
                var image=Path.GetFullPath(Path.IsPathRooted(frame.image_path)?frame.image_path:Path.Combine(Path.GetDirectoryName(path),frame.image_path));
                candidates.Add(new MageInputFrame{frame_id="v"+i.ToString("D3",CultureInfo.InvariantCulture),timestamp_ms=frame.timestamp_ms,image_path=image});
            }
            var selected=candidates.Skip(Math.Max(0,candidates.Count-window)).ToArray();if(selected.Length==0)throw new ArgumentException("No preview frames before at_ms.");
            var freshTextures=new List<Texture2D>();
            try
            {
                foreach(var frame in selected)
                {
                    if(new FileInfo(frame.image_path).Length>20*1024*1024)throw new IOException("Image exceeds 20 MB.");
                    frame.sha256=FileSha256(frame.image_path);var texture=new Texture2D(2,2,TextureFormat.RGB24,false);freshTextures.Add(texture);
                    if(!texture.LoadImage(File.ReadAllBytes(frame.image_path)))throw new IOException("Cannot decode preview image.");texture.name=frame.frame_id+"@"+frame.sha256;
                }
            }
            catch{foreach(var texture in freshTextures)Destroy(texture);throw;}
            foreach(var texture in textures)Destroy(texture);textures.Clear();textures.AddRange(freshTextures);previewFrames=selected;
            for(int i=0;i<thumbnails.Count;i++)
            {
                thumbnails[i].enabled=i<textures.Count;thumbnails[i].texture=i<textures.Count?textures[i]:null;
                thumbnailLabels[i].text=i<selected.Length?selected[i].frame_id+"\n"+(selected[i].timestamp_ms/1000.0).ToString("F1",CultureInfo.InvariantCulture)+" s":"—";
                if(i<textures.Count)Fit(thumbnails[i],textures[i],3,3,76,43);
            }
            SelectFrame(selected.Length-1);
        }
        private static void Fit(RawImage image,Texture texture,float x,float y,float width,float height)
        {
            float scale=Mathf.Min(width/texture.width,height/texture.height);var size=new Vector2(texture.width*scale,texture.height*scale);
            var rect=image.rectTransform;rect.anchorMin=rect.anchorMax=rect.pivot=new Vector2(0,1);rect.sizeDelta=size;rect.anchoredPosition=new Vector2(x+(width-size.x)/2,-(y+(height-size.y)/2));
        }
        public void SelectFrame(int index)
        {
            if(index<0||index>=textures.Count)return;SelectedFrame=index;mainImage.texture=textures[index];Fit(mainImage,textures[index],24,48,712,258);
            var frame=previewFrames[index];mainCaption.text=frame.frame_id+"   /   "+(frame.timestamp_ms/1000.0).ToString("F1",CultureInfo.InvariantCulture)+" s   /   "+previewFrames.Length+" 帧窗口";
            for(int i=0;i<thumbnailLabels.Count;i++)thumbnailLabels[i].color=i==index?mint:muted;
        }
        private void SetIdle()
        {
            RequestState="idle";ClearAnswer("尚未请求模型。左侧仅为本地输入预览。");statusText.text="选择缩略图检查输入，点击按钮发起一次真实 Mage-VL 请求。";
        }
        private void ClearAnswer(string detail)
        {
            SetScroll(detailsText,detailsContent,detail+"\n\n传感器：无 IMU。没有手腕或头部传感器输入。",310);
            SetScroll(answerText,answerContent,"",118);
        }
        public void Ask()
        {
            if(!initialized||RequestState=="loading")return;
            LastResult=null;LastResponseJson=null;LastRequest=null;RequestFile=null;ResponseFile=null;HelperExitCode=null;ResponseAcceptedFromOwnProcess=false;
            RequestId=Guid.NewGuid().ToString();RequestState="loading";ClearAnswer("正在等待本次模型回答；未沿用上次结果。");
            askButton.interactable=false;questionInput.interactable=false;askLabel.text="请求中…";runText.text="";
            summaryText.text="模型 Mage-VL · 请求中 · 无 IMU 输入";statusText.text="请求 "+RequestId+" · 正在启动本地桥接";
            try
            {
                LoadPreview();requestedFrames=previewFrames.Select(f=>new MageInputFrame{frame_id=f.frame_id,timestamp_ms=f.timestamp_ms,image_path=f.image_path,sha256=f.sha256}).ToArray();
                var question=QuestionText.Trim();if(string.IsNullOrWhiteSpace(question))throw new ArgumentException("Question is blank.");
                LastRequest=new MageRequest{request_id=RequestId,question=question,session_path=SessionAbsolute(),at_ms=atMs,window=window};
                var directory=Path.Combine(ProjectRoot,"UserSettings/SatoriMage");Directory.CreateDirectory(directory);
                RequestFile=Path.Combine(directory,RequestId+".request.json");ResponseFile=Path.Combine(directory,RequestId+".response.json");
                if(File.Exists(ResponseFile))throw new IOException("Response path unexpectedly exists.");
                using(var stream=new FileStream(RequestFile,FileMode.CreateNew,FileAccess.Write,FileShare.Read))using(var writer=new StreamWriter(stream,new UTF8Encoding(false)))writer.Write(MageJson.Write(LastRequest));
                var python=ResolvePython();var bridge=Path.Combine(ProjectRoot,"Tools/unity_mage_request.py");if(!File.Exists(bridge))throw new FileNotFoundException("Bridge script is missing.");
                stdoutCharacters=stderrCharacters=0;
                helper=new Process{StartInfo=new ProcessStartInfo{FileName=python,Arguments=Quote(bridge)+" --request "+Quote(RequestFile)+" --response "+Quote(ResponseFile),WorkingDirectory=ProjectRoot,UseShellExecute=false,CreateNoWindow=true,RedirectStandardOutput=true,RedirectStandardError=true}};
                helper.OutputDataReceived+=(sender,e)=>{if(e.Data!=null)Interlocked.Add(ref stdoutCharacters,e.Data.Length);};
                helper.ErrorDataReceived+=(sender,e)=>{if(e.Data!=null)Interlocked.Add(ref stderrCharacters,e.Data.Length);};
                if(!helper.Start())throw new InvalidOperationException("Helper did not start.");helper.BeginOutputReadLine();helper.BeginErrorReadLine();helperStarted=Time.realtimeSinceStartup;
            }
            catch(Exception){StopOwnHelper();Fail("client_start_failed","本地请求未能启动，请检查 Python、桥接脚本、素材路径与问题。",null);}
        }
        private string ResolvePython()
        {
            var path=pythonExecutableOverride;
            if(string.IsNullOrWhiteSpace(path))
            {
                path=Path.Combine(ProjectRoot,".venv/Scripts/python.exe");
                if(!File.Exists(path))path=Path.Combine(ProjectRoot,"Research/Backend/.venv/Scripts/python.exe");
            }
            path=Path.GetFullPath(Path.IsPathRooted(path)?path:Path.Combine(ProjectRoot,path));if(!File.Exists(path))throw new FileNotFoundException("Python executable is missing.");return path;
        }
        private static string Quote(string argument)
        {
            var output=new StringBuilder("\"");int slash=0;
            foreach(char c in argument)
            {
                if(c=='\\'){slash++;continue;}
                if(c=='\"'){output.Append('\\',slash*2+1);output.Append(c);}else{output.Append('\\',slash);output.Append(c);}slash=0;
            }
            output.Append('\\',slash*2);output.Append('"');return output.ToString();
        }
        private void Update()
        {
            if(RequestState!="loading"||helper==null)return;
            try
            {
                if(helper.HasExited){HelperExitCode=helper.ExitCode;helper.Dispose();helper=null;AcceptResponse();return;}
                var elapsed=Time.realtimeSinceStartup-helperStarted;statusText.text="请求 "+RequestId+" · 等待真实模型结果 "+elapsed.ToString("F0")+" s";
                if(elapsed>helperTimeoutSeconds){StopOwnHelper();Fail("client_timeout","等待超时；本地助手已停止，可重试。没有终止模型服务器。",null);}
            }
            catch(Exception){StopOwnHelper();Fail("client_process_error","本地桥接进程状态异常，请查看本次请求文件。",null);}
        }
        private void StopOwnHelper()
        {
            if(helper==null)return;
            try{if(!helper.HasExited)helper.Kill();}catch(InvalidOperationException){}catch(System.ComponentModel.Win32Exception){}
            finally{helper.Dispose();helper=null;}
        }
        public static bool CanPresentResult(string state,MageResponse response,string requestId)
        {
            return state=="completed"&&response!=null&&response.status=="completed"&&response.request_id==requestId&&response.answer_payload!=null&&!string.IsNullOrWhiteSpace(response.raw_answer_json);
        }
        private void AcceptResponse()
        {
            MageResponse response=null;
            try
            {
                if(!File.Exists(ResponseFile)){Fail("response_missing","桥接进程已退出，但没有生成本次响应文件。",null);return;}
                LastResponseJson=MageJson.ReadFile(ResponseFile);response=MageJson.Read<MageResponse>(LastResponseJson);
                if(response==null||response.request_id!=RequestId||response.question!=LastRequest.question)throw new InvalidDataException("Response identity mismatch.");
                if(response.status!="completed"){Fail(response.error_code??"bridge_failed",response.message??"模型请求失败。",response.run_directory);return;}
                if(HelperExitCode!=0)throw new InvalidDataException("Helper exit code disagrees with success.");
                ValidateResponse(response);
                LastResult=response;RequestState="completed";ResponseAcceptedFromOwnProcess=true;CompletedRequestCount++;
                if(!CanPresentResult(RequestState,LastResult,RequestId))throw new InvalidDataException("Response cannot be presented.");
                var answer=response.answer_payload;
                SetScroll(detailsText,detailsContent,"环境："+answer.environment.place_type+"\n\n可见物体："+string.Join("、",answer.environment.visible_objects??Array.Empty<string>())+"\n\n动作："+answer.activity.action+"\n阶段："+answer.activity.phase+"\n\n不确定性：\n"+string.Join("\n",answer.uncertainties??Array.Empty<string>())+"\n\n视觉证据："+string.Join(" · ",answer.evidence_frame_ids)+"\n传感器：无 IMU；不推断头部或手腕惯性数据。",310);
                SetScroll(answerText,answerContent,response.raw_answer_json,118);
                summaryText.text="模型 "+response.model+" · "+(response.latency_ms/1000.0).ToString("F2",CultureInfo.InvariantCulture)+" s · "+response.input_frames.Length+" 帧 · 无 IMU 输入";
                statusText.text="本次请求完成 · "+RequestId+" · 仅作真实模型问答测试，不是准确率评估";runText.text="run："+response.run_directory;EnableRetry();
            }
            catch(Exception){Fail("response_validation_failed","响应身份、输入图片或证据校验失败；未展示为成功。",response?.run_directory);}
        }
        private void ValidateResponse(MageResponse response)
        {
            if(string.IsNullOrWhiteSpace(response.model)||string.IsNullOrWhiteSpace(response.run_directory)||response.input_frames==null||response.input_frames.Length!=requestedFrames.Length)throw new InvalidDataException("Incomplete response.");
            if(double.IsNaN(response.latency_ms)||double.IsInfinity(response.latency_ms)||response.latency_ms<0)throw new InvalidDataException("Invalid latency.");
            for(int i=0;i<requestedFrames.Length;i++)
            {
                var expected=requestedFrames[i];var actual=response.input_frames[i];
                if(actual==null||actual.frame_id!=expected.frame_id||actual.timestamp_ms!=expected.timestamp_ms||actual.timestamp_ms>LastRequest.at_ms||
                    !string.Equals(actual.sha256,expected.sha256,StringComparison.OrdinalIgnoreCase)||
                    !string.Equals(Path.GetFullPath(actual.image_path),expected.image_path,StringComparison.OrdinalIgnoreCase)||
                    FileSha256(actual.image_path)!=expected.sha256)throw new InvalidDataException("Input frame mismatch.");
            }
            var answer=response.answer_payload;
            if(answer?.environment==null||answer.activity==null||string.IsNullOrWhiteSpace(answer.answer)||string.IsNullOrWhiteSpace(response.raw_answer_json))throw new InvalidDataException("Missing answer.");
            var parsed=MageJson.Read<MageAnswer>(response.raw_answer_json);
            if(MageJson.Write(parsed)!=MageJson.Write(answer))throw new InvalidDataException("Raw answer does not match payload.");
            var allowed=new HashSet<string>(requestedFrames.Select(f=>f.frame_id));
            if(answer.evidence_frame_ids==null||answer.evidence_frame_ids.Length==0||answer.evidence_frame_ids.Any(id=>!allowed.Contains(id)))throw new InvalidDataException("Invalid evidence.");
            if(answer.sensor_evidence_ids==null||answer.sensor_evidence_ids.Length!=0||response.missing_sensors==null||!response.missing_sensors.Contains("imu"))throw new InvalidDataException("Sensor provenance mismatch.");
        }
        private void EnableRetry(){askButton.interactable=true;questionInput.interactable=true;askLabel.text="询问 Mage-VL";}
        private void Fail(string code,string message,string run)
        {
            RequestState="failed";ResponseAcceptedFromOwnProcess=false;
            LastResult=new MageResponse{status="failed",request_id=RequestId,question=LastRequest?.question,error_code=code,message=message,run_directory=run,input_frames=Array.Empty<MageInputFrame>(),missing_sensors=new[]{"imu"}};
            if(!initialized)return;
            ClearAnswer("本次请求失败，未展示任何旧回答。");summaryText.text="模型 Mage-VL · 本次未成功 · 无 IMU 输入";
            statusText.text=code+"："+message;runText.text=string.IsNullOrWhiteSpace(run)?"请求文件："+(RequestFile??"尚未创建"):"失败 run："+run;EnableRetry();
        }
    }
}
