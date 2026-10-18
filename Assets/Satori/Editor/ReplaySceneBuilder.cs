using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace SatoriReplay.Editor
{
    public static class ReplaySceneBuilder
    {
        public const string ScenePath="Assets/Satori/Scenes/SatoriReplay.unity";
        [MenuItem("Satori/Create Replay Scene")]
        public static void CreateScene()
        {
            var scene=EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
            var camera=new GameObject("Replay Camera").AddComponent<Camera>();camera.clearFlags=CameraClearFlags.SolidColor;camera.backgroundColor=new Color(0.035f,0.055f,0.065f);
            var hud=new GameObject("Satori Replay HUD").AddComponent<ReplayHud>();hud.demoReport=AssetDatabase.LoadAssetAtPath<TextAsset>("Assets/Satori/Data/demo-report.json");
            Directory.CreateDirectory("Assets/Satori/Scenes");EditorSceneManager.SaveScene(scene,ScenePath);
            Debug.Log("SATORI_SCENE_READY "+ScenePath);
        }
        [MenuItem("Satori/Build Windows Replay")]
        public static void BuildWindows()
        {
            if(!File.Exists(ScenePath))CreateScene();
            var directory=Path.GetFullPath("Builds/Windows");Directory.CreateDirectory(directory);
            var report=BuildPipeline.BuildPlayer(new[]{ScenePath},Path.Combine(directory,"SatoriReplay.exe"),BuildTarget.StandaloneWindows64,BuildOptions.None);
            Directory.CreateDirectory("Assets/Satori/Evidence");
            File.WriteAllText("Assets/Satori/Evidence/build-result.json","{\"result\":\""+report.summary.result+"\",\"errors\":"+report.summary.totalErrors+",\"warnings\":"+report.summary.totalWarnings+"}");
            Debug.Log("SATORI_BUILD_RESULT "+report.summary.result+" "+directory);
        }
    }
}
