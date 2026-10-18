using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

namespace SatoriReplay.Editor
{
    public static class RealDataPlayChecks
    {
        [Serializable] private sealed class FrameProof
        {
            public int index;public string frame_id;public string image_path;
            public int image_width;public int image_height;public string[] evidence_frame_ids;
        }
        [Serializable] private sealed class CheckResult
        {
            public bool success;public int passed;public string failure;public string unity_version;
            public string session_id;public bool synthetic;public string source_kind;public string source_label;
            public string report_path;public string report_sha256;public string provider;public string model;
            public FrameProof[] frames;public string[] checks;
        }
        [MenuItem("Satori/Run Real Data Checks")]
        public static void Run()
        {
            var result=new CheckResult{unity_version=Application.unityVersion};
            var names=new List<string>();var proofs=new List<FrameProof>();
            var hud=UnityEngine.Object.FindObjectOfType<ReplayHud>();int original=hud!=null?hud.CurrentFrame:-1;
            Action<bool,string> check=(ok,name)=>{if(!ok)throw new Exception(name);names.Add(name);};
            try
            {
                check(EditorApplication.isPlaying,"Editor is in Play Mode");
                check(hud!=null&&hud.LoadedReport!=null&&hud.CurrentFrame>=0,"HUD loaded a report");
                var report=hud.LoadedReport;result.report_path=hud.LoadedReportPath;result.session_id=report.session_id;
                result.synthetic=report.synthetic;result.source_kind=report.source_kind;result.source_label=report.source_label;
                check(!report.synthetic,"Report is non-synthetic");
                check(File.Exists(result.report_path),"Report file exists");
                var disk=ReportParser.Load(result.report_path);
                check(!disk.synthetic&&disk.session_id==report.session_id&&disk.records.Length==report.records.Length,"Loaded report agrees with source file");
                using(var sha=SHA256.Create())using(var input=File.OpenRead(result.report_path))result.report_sha256=BitConverter.ToString(sha.ComputeHash(input)).Replace("-","").ToLowerInvariant();
                var manifest=ModelRunLoader.ReadManifest(Path.Combine(Path.GetDirectoryName(result.report_path),"run.json"));
                result.provider=manifest.provider;result.model=manifest.model;
                check(ModelRunLoader.IsModelRun(manifest)&&manifest.session_id==report.session_id,"Completed model manifest matches report");
                check(!string.IsNullOrWhiteSpace(report.source_kind)&&!string.IsNullOrWhiteSpace(report.source_label),"Dataset/input provenance is explicitly declared");
                var texts=hud.GetComponentsInChildren<Text>();
                check(texts.Any(t=>t.name=="Provenance"&&t.text.Contains("非合成数据")),"Non-synthetic provenance is visible");
                check(texts.Any(t=>t.name=="Mode"&&t.text.Contains("未连接实时相机")),"UI does not claim live glasses capture");
                var kind=report.source_kind.ToLowerInvariant();
                if(kind.Contains("public")||kind.Contains("dataset"))check(texts.Any(t=>t.name=="Source"&&t.text.Contains("非用户实拍")),"Public video is distinguished from user footage");
                else check(texts.Any(t=>t.name=="Source"&&t.text.Contains(report.source_label)),"Input source label is displayed");
                var selected=new[]{0,report.records.Length/2,report.records.Length-1}.Distinct().ToArray();
                foreach(var index in selected)
                {
                    hud.SeekFrame(index);var record=report.records[index];var expected=disk.records[index];
                    check(hud.CurrentFrame==index&&hud.CurrentRecord.frame_id==expected.frame_id&&record.timestamp_ms==expected.timestamp_ms,"Current frame matches report at "+index);
                    check(record.evidence_frame_ids.SequenceEqual(expected.evidence_frame_ids),"Evidence IDs match report at "+index);
                    check(record.evidence_frame_ids.Length>0&&record.evidence_frame_ids.All(id=>report.records.Take(index+1).Any(r=>r.frame_id==id)),"Evidence is present and never references a future frame at "+index);
                    var references=texts.Single(t=>t.name=="References");
                    check(references.text=="证据："+string.Join(" · ",record.evidence_frame_ids),"Displayed evidence matches report at "+index);
                    check(texts.Single(t=>t.name=="Frame").text.StartsWith(record.frame_id+"  /  "),"Displayed frame ID matches selected record at "+index);
                    check(!string.IsNullOrWhiteSpace(record.image_path),"Image path is provided at "+index);
                    var path=Path.IsPathRooted(record.image_path)?record.image_path:Path.Combine(Path.GetDirectoryName(result.report_path),record.image_path);
                    check(File.Exists(path),"Referenced image exists at "+index);
                    var raw=hud.GetComponentsInChildren<RawImage>().Single();
                    check(hud.HasLoadedImage&&raw.enabled&&raw.texture==hud.CurrentImage&&raw.texture.width>0&&raw.texture.height>0,"Real image texture is loaded and displayed at "+index);
                    check(!texts.Any(t=>t.name=="Image Note"&&t.gameObject.activeInHierarchy),"Missing-image placeholder is hidden at "+index);
                    proofs.Add(new FrameProof{index=index,frame_id=record.frame_id,image_path=Path.GetFullPath(path),image_width=raw.texture.width,image_height=raw.texture.height,evidence_frame_ids=record.evidence_frame_ids});
                }
                result.success=true;Debug.Log("SATORI_REAL_DATA_CHECKS_PASS "+names.Count);
            }
            catch(Exception error){result.success=false;result.failure=error.Message;Debug.LogError("SATORI_REAL_DATA_CHECKS_FAILED "+error.Message);}
            finally
            {
                if(hud!=null&&hud.LoadedReport!=null&&original>=0)hud.SeekFrame(original);
                result.passed=names.Count;result.frames=proofs.ToArray();result.checks=names.ToArray();
                Directory.CreateDirectory("Assets/Satori/Evidence");File.WriteAllText("Assets/Satori/Evidence/real-data-tests.json",JsonUtility.ToJson(result,true));
            }
            if(!result.success)throw new InvalidOperationException(result.failure);
        }
    }
}
