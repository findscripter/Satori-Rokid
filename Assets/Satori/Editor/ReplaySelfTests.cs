using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace SatoriReplay.Editor
{
    public static class ReplaySelfTests
    {
        private const string Valid="{\"session_id\":\"test\",\"synthetic\":true,\"records\":[{\"frame_id\":\"f1\",\"timestamp_ms\":0,\"confidence\":0.9,\"decision_kind\":\"observe\"},{\"frame_id\":\"f2\",\"timestamp_ms\":1000,\"confidence\":0.8,\"decision_kind\":\"guide\"}]}";
        private static int passed;
        private static void Check(bool value,string name){if(!value)throw new Exception("FAIL: "+name);passed++;Debug.Log("SATORI_CHECK_PASS "+name);}
        private static void Reject(string json,string name){try{ReportParser.Parse(json);}catch(ArgumentException){passed++;Debug.Log("SATORI_CHECK_PASS "+name);return;}throw new Exception("FAIL accepted invalid data: "+name);}
        [MenuItem("Satori/Run Core Checks")]
        public static void Run()
        {
            passed=0;
            try
            {
                var report=ReportParser.Parse(Valid);Check(report.records.Length==2&&report.synthetic,"parse synthetic report");
                Check(report.records[0].confirmed_completed_step_ids.Length==0,"optional arrays normalize");
                Reject("{}","missing session and records");Reject(Valid.Replace("\"synthetic\":true,",""),"missing provenance declaration");
                Reject(Valid.Replace("\"f2\"","\"f1\""),"duplicate frames");Reject(Valid.Replace("1000","0"),"timestamps increase strictly");
                Reject(Valid.Replace("0.9","1.9"),"confidence range");Reject(Valid.Replace("\"guide\"","\"unknown\""),"decision vocabulary");
                var player=new ReplayPlayback(report);player.Previous();Check(player.Index==0,"previous at first frame");
                player.Next();player.Next();Check(player.Index==1,"next at last frame");
                player.Toggle();Check(player.Index==0&&player.Playing,"restart from end");
                player.Advance(0.4);Check(player.Index==0,"wait for recorded timestamp");
                player.Advance(0.6);Check(player.Index==1&&!player.Playing,"stop at end");
                player.Seek(0);player.Speed=2;player.Toggle();player.Advance(0.5);Check(player.Index==1,"speed scales recorded time");
                player.Seek(-4);Check(player.Index==0,"seek lower boundary");player.Seek(99);Check(player.Index==1,"seek upper boundary");
                Reject("{\"session_id\":\"x\",\"synthetic\":true,\"records\":[]}","empty timeline rejects");
                Reject(Valid.Replace("\"decision_kind\":\"guide\"","\"decision_kind\":\"guide\",\"evidence_frame_ids\":[\"future\"]"),"future evidence rejects");
                var single=ReportParser.Parse("{\"session_id\":\"x\",\"synthetic\":false,\"records\":[{\"frame_id\":\"f\",\"timestamp_ms\":30,\"confidence\":0,\"decision_kind\":\"clarify\"}]}");
                var one=new ReplayPlayback(single);one.Toggle();Check(!one.Playing&&one.Index==0,"single-frame boundary");
                var path=File.Exists("Assets/Satori/Data/demo-report.json")?"Assets/Satori/Data/demo-report.json":"Assets/StreamingAssets/demo-report.json";
                var demo=ReportParser.Load(path);Check(demo.records.Length==9&&demo.synthetic,"backend nine-frame synthetic fixture");
                Check(demo.records[2].uncertain&&demo.records[2].confidence<0.5,"backend occlusion uncertainty retained");
                Check(demo.records[8].decision_kind=="complete"&&demo.records[8].confirmed_completed_step_ids.Length>0,"backend completion has confirmation");
                Save(0);Debug.Log("SATORI_CORE_TESTS_PASS "+passed);
            }
            catch(Exception error){Save(1);Debug.LogError("SATORI_CORE_TESTS_FAILED after "+passed+" checks: "+error);throw;}
        }
        private static void Save(int failures)
        {
            Directory.CreateDirectory("Assets/Satori/Evidence");
            File.WriteAllText("Assets/Satori/Evidence/core-tests.json","{\"passed\":"+passed+",\"failed\":"+failures+",\"unity_version\":\""+Application.unityVersion+"\"}");
        }
    }
}
