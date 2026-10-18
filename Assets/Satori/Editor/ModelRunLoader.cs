using System;
using System.Globalization;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace SatoriReplay.Editor
{
    [DataContract] public sealed class ModelRunManifest
    {
        [DataMember(IsRequired=true)] public string status;
        [DataMember(IsRequired=true)] public bool synthetic;
        [DataMember(IsRequired=true)] public string session_id;
        [DataMember] public string provider;
        [DataMember] public string model;
        [DataMember] public string finished_utc;
        [DataMember] public int processed_frames;
    }
    public static class ModelRunLoader
    {
        public static ModelRunManifest ReadManifest(string path)
        {
            using(var stream=File.OpenRead(path))return (ModelRunManifest)new DataContractJsonSerializer(typeof(ModelRunManifest)).ReadObject(stream);
        }
        public static bool IsModelRun(ModelRunManifest manifest)
        {
            return manifest!=null&&manifest.status=="completed"&&!manifest.synthetic&&
                !string.IsNullOrWhiteSpace(manifest.provider)&&manifest.provider!="replay";
        }
        public static string LatestReport(string runsRoot)
        {
            if(!Directory.Exists(runsRoot))throw new DirectoryNotFoundException("模型 runs 目录不存在："+runsRoot);
            string chosen=null;DateTimeOffset newest=DateTimeOffset.MinValue;
            foreach(var directory in Directory.GetDirectories(runsRoot))
            {
                var manifestPath=Path.Combine(directory,"run.json");var reportPath=Path.Combine(directory,"unity-report.json");
                if(!File.Exists(manifestPath)||!File.Exists(reportPath))continue;
                try
                {
                    var manifest=ReadManifest(manifestPath);if(!IsModelRun(manifest))continue;
                    var report=ReportParser.Load(reportPath);
                    if(report.synthetic||report.session_id!=manifest.session_id||
                        (manifest.processed_frames>0&&manifest.processed_frames!=report.records.Length))continue;
                    if(!DateTimeOffset.TryParse(manifest.finished_utc,CultureInfo.InvariantCulture,DateTimeStyles.RoundtripKind,out var finished))continue;
                    if(finished>newest||(finished==newest&&string.CompareOrdinal(reportPath,chosen)>0)){newest=finished;chosen=reportPath;}
                }
                catch(Exception e){Debug.LogWarning("Satori skipped invalid run "+Path.GetFileName(directory)+": "+e.Message);}
            }
            if(chosen==null)throw new InvalidOperationException("未发现 completed、synthetic=false 且 provider 非 replay 的有效模型报告；保留当前回放。");
            return Path.GetFullPath(chosen);
        }
        [MenuItem("Satori/Load Latest Model Run")]
        public static void LoadLatest()
        {
            var root=Path.GetFullPath(Path.Combine(Application.dataPath,"../Research/Backend/runs"));
            var path=LatestReport(root);
            var hud=UnityEngine.Object.FindObjectOfType<ReplayHud>();
            if(hud==null)throw new InvalidOperationException("请先打开 SatoriReplay 场景。");
            if(!EditorApplication.isPlaying)Undo.RecordObject(hud,"Load Satori Model Run");
            hud.LoadReport(path);
            if(!EditorApplication.isPlaying){EditorUtility.SetDirty(hud);EditorSceneManager.MarkSceneDirty(hud.gameObject.scene);}
            Debug.Log("SATORI_MODEL_REPORT_SELECTED "+path);
        }
    }
}
