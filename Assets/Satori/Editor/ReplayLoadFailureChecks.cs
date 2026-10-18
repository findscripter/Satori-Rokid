using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;

namespace SatoriReplay.Editor
{
    // Actual Unity UI regression checks. Fixtures are local and never call a model.
    public static class ReplayLoadFailureChecks
    {
        private const string Output="Evidence/code-fixes-20261017/unity-reload-checks.json";
        private const string SceneOutput="Evidence/code-fixes-20261017/unity-default-scene-checks.json";
        private static readonly BindingFlags Private=BindingFlags.Instance|BindingFlags.NonPublic;
        [Serializable] private sealed class Result
        {
            public string unity_version=Application.unityVersion;
            public int passed,failed;
            public List<string> checks=new List<string>(),failures=new List<string>();
        }
        private static void Check(Result result,bool ok,string name)
        {
            result.checks.Add(name);
            if(ok)result.passed++;else{result.failed++;result.failures.Add(name);}
        }
        private static void Save(Result result,string path)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path,JsonUtility.ToJson(result,true));
            if(result.failed==0)Debug.Log("SATORI_LOAD_REGRESSION_PASS "+path+" "+result.passed);
            else Debug.LogError("SATORI_LOAD_REGRESSION_FAIL "+path+" "+string.Join("; ",result.failures));
        }
        [MenuItem("Satori/Run Default Replay Entry Checks")]
        public static void CheckDefaultScene()
        {
            if(EditorApplication.isPlaying)throw new InvalidOperationException("Check the default entry before Play Mode.");
            var result=new Result();
            var existing=UnityEngine.SceneManagement.SceneManager.GetSceneByPath(ReplaySceneBuilder.ScenePath);
            bool wasLoaded=existing.IsValid()&&existing.isLoaded;
            if(wasLoaded&&existing.isDirty)throw new InvalidOperationException("Save or discard scene edits before checking the saved default entry.");
            var preview=wasLoaded?existing:EditorSceneManager.OpenScene(ReplaySceneBuilder.ScenePath,OpenSceneMode.Additive);
            try
            {
                var roots=preview.GetRootGameObjects();
                Check(result,roots.SelectMany(r=>r.GetComponentsInChildren<ReplayHud>(true)).Any(h=>h.gameObject.activeSelf&&h.enabled),"Saved scene starts the replay HUD");
                Check(result,!roots.SelectMany(r=>r.GetComponentsInChildren<MageContextHud>(true)).Any(h=>h.gameObject.activeSelf&&h.enabled),"Saved scene leaves the optional Mage view inactive");
            }
            finally{if(!wasLoaded)EditorSceneManager.CloseScene(preview,true);Save(result,SceneOutput);}
        }
        private static string Text(ReplayHud hud,string name)=>hud.GetComponentsInChildren<Text>(true).Single(t=>t.name==name).text;
        private static void CheckEmpty(Result result,ReplayHud hud,string prefix)
        {
            Check(result,hud.LoadedReport==null&&hud.CurrentRecord==null&&hud.CurrentFrame==-1&&!hud.IsPlaying,prefix+": no loaded report or playing state");
            Check(result,hud.LoadedReportPath==null,prefix+": loaded path cleared");
            Check(result,hud.CurrentImage==null&&!hud.HasLoadedImage,prefix+": old image detached");
            var picture=hud.GetComponentInChildren<RawImage>(true);
            Check(result,picture.texture==null&&!picture.enabled,prefix+": raw image cleared and hidden");
            foreach(var name in new[]{"Task","Scene","Action","Step","Confidence","Confirmed","Observed","Decision","References","Frame","Source"})
                Check(result,Text(hud,name)=="暂无记录",prefix+": "+name+" resets to an empty state");
            var slider=hud.GetComponentInChildren<Slider>(true);
            Check(result,slider.value==0&&!slider.interactable,prefix+": empty timeline reset and disabled");
            Check(result,hud.GetComponentsInChildren<Button>(true).Where(b=>b.name!="重新读取").All(b=>!b.interactable),prefix+": playback controls disabled");
            Check(result,hud.GetComponentsInChildren<Text>(true).Any(t=>t.text=="报告未加载"),prefix+": error status visible");
            var note=hud.GetComponentsInChildren<Text>(true).Single(t=>t.name=="Image Note");
            Check(result,note.gameObject.activeSelf&&!string.IsNullOrWhiteSpace(note.text),prefix+": error reason visible");
            // Destroy is deferred in Play Mode; native resource disposal is checked after a frame.
        }
        [MenuItem("Satori/Run Report Load Failure Checks")]
        public static void Run()
        {
            if(!EditorApplication.isPlaying)throw new InvalidOperationException("Run report failure checks in Play Mode.");
            var result=new Result();var oldImages=new List<Texture2D>();
            string folder=Path.Combine(Path.GetTempPath(),"satori-reload-"+Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(folder);
            var root=new GameObject("Satori load regression fixture");
            var hud=root.AddComponent<ReplayHud>();hud.enabled=false;
            try
            {
                // Disabled fixture prevents Unity Start from executing twice.
                typeof(ReplayHud).GetMethod("Start",Private).Invoke(hud,null);
                var image=new Texture2D(2,2);image.SetPixels(new[]{Color.red,Color.green,Color.blue,Color.white});image.Apply();
                File.WriteAllBytes(Path.Combine(folder,"input.png"),image.EncodeToPNG());UnityEngine.Object.Destroy(image);
                var record="{\"frame_id\":\"FRAME\",\"timestamp_ms\":TIME,\"task_title\":\"OLD_TASK\",\"scene\":\"OLD_SCENE\",\"action\":\"OLD_ACTION\",\"current_step_id\":\"OLD_STEP\",\"current_step_instruction\":\"OLD_INSTRUCTION\",\"confidence\":0.91,\"confirmed_completed_step_ids\":[\"OLD_CONFIRMED\"],\"observed_completed_step_ids\":[\"OLD_OBSERVED\"],\"decision_kind\":\"guide\",\"decision_message\":\"OLD_GUIDANCE\",\"evidence_frame_ids\":[\"FRAME\"],\"image_path\":\"input.png\"}";
                string valid=Path.Combine(folder,"valid.json"),missing=Path.Combine(folder,"missing.json"),bad=Path.Combine(folder,"malformed.json");
                File.WriteAllText(valid,"{\"session_id\":\"regression\",\"synthetic\":true,\"records\":["+record.Replace("FRAME","f1").Replace("TIME","0")+","+record.Replace("FRAME","f2").Replace("TIME","1000")+"]}");
                File.WriteAllText(bad,"{invalid JSON}");
                void LoadPopulated()
                {
                    hud.LoadReport(valid);hud.SeekFrame(1);hud.TogglePlayback();
                    Check(result,hud.HasLoadedImage&&Text(hud,"Scene")=="OLD_SCENE"&&hud.IsPlaying,"Fixture loads image, labels and playback");
                    oldImages.Add(hud.CurrentImage);
                }
                LoadPopulated();hud.reportPathOverride=missing;hud.ReloadReport();
                CheckEmpty(result,hud,"Reload missing");
                LoadPopulated();hud.reportPathOverride=bad;hud.ReloadReport();
                CheckEmpty(result,hud,"Reload malformed");
                foreach(var path in new[]{missing,bad,""})
                {
                    LoadPopulated();bool rejected=false;
                    try{hud.LoadReport(path);}catch(Exception){rejected=true;}
                    Check(result,rejected,"LoadReport invalid input still rejects");
                    CheckEmpty(result,hud,"Direct load "+Path.GetFileName(path));
                }
                hud.LoadReport(valid);
                Check(result,hud.HasLoadedImage&&hud.GetComponentInChildren<Slider>(true).interactable,"Successful load after errors restores image and timeline");
                Check(result,hud.GetComponentsInChildren<Button>(true).All(b=>b.interactable),"Successful load restores playback controls");
            }
            catch(Exception error){result.failed++;result.failures.Add("Unexpected test exception: "+error);}
            double earliest=EditorApplication.timeSinceStartup+0.25;
            void Finish()
            {
                if(EditorApplication.timeSinceStartup<earliest)return;
                EditorApplication.update-=Finish;
                Check(result,oldImages.All(t=>t==null),"Superseded evidence textures are destroyed after a player frame");
                UnityEngine.Object.Destroy(root);
                foreach(var file in new[]{"input.png","valid.json","malformed.json"})File.Delete(Path.Combine(folder,file));
                Directory.Delete(folder);
                Save(result,Output);
            }
            EditorApplication.update+=Finish;
        }
    }
}
