using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

namespace SatoriReplay.Editor
{
    public static class ReplayPlayChecks
    {
        [MenuItem("Satori/Run Play Mode Checks")]
        public static void Run()
        {
            if(!EditorApplication.isPlaying)throw new InvalidOperationException("请先进入 Play Mode 并等待一帧。");
            int passed=0;
            Action<bool,string> check=(ok,name)=>{if(!ok)throw new Exception("SATORI_PLAY_FAIL "+name);passed++;Debug.Log("SATORI_PLAY_PASS "+name);};
            var hud=UnityEngine.Object.FindObjectOfType<ReplayHud>();
            check(hud!=null&&hud.CurrentFrame>=0,"HUD loaded report");
            var texts=hud.GetComponentsInChildren<Text>();
            check(texts.Any(t=>t.text.Contains("合成数据")),"synthetic provenance visible");
            check(texts.Any(t=>t.text.Contains("未连接实时相机")),"no realtime claim");
            for(int i=0;i<20;i++)hud.PreviousFrame();
            check(hud.CurrentFrame==0,"previous clamps at start");
            var next=hud.GetComponentsInChildren<Button>().Single(b=>b.name=="下一条");next.onClick.Invoke();
            check(hud.CurrentFrame==1,"next UI button changes frame");
            check(texts.Any(t=>t.name=="References"&&t.text.Contains("f1")),"evidence changes with frame");
            var previous=hud.GetComponentsInChildren<Button>().Single(b=>b.name=="上一条");previous.onClick.Invoke();
            check(hud.CurrentFrame==0,"previous UI button changes frame");
            hud.TogglePlayback();check(hud.IsPlaying,"play starts");hud.TogglePlayback();check(!hud.IsPlaying,"pause stops");
            var font=texts.First(t=>t.font!=null).font;const string chinese="环境任务回放合成数据用户确认观察辅助决策证据";
            font.RequestCharactersInTexture(chinese,24,FontStyle.Normal);
            check(chinese.All(c=>font.HasCharacter(c)),"Chinese glyphs available in selected runtime font");
            Directory.CreateDirectory("Assets/Satori/Evidence");
            File.WriteAllText("Assets/Satori/Evidence/play-tests.json","{\"passed\":"+passed+",\"failed\":0,\"unity_version\":\""+Application.unityVersion+"\",\"font\":\""+font.name+"\"}");
            Debug.Log("SATORI_PLAY_TESTS_PASS "+passed);
        }
    }
}
