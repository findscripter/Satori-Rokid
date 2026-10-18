using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace SatoriReplay.Editor
{
    public static class MageContextMenus
    {
        [MenuItem("Satori/Open Mage-VL Test")]
        public static void Open()
        {
            var scene=UnityEngine.SceneManagement.SceneManager.GetActiveScene();
            var hud=Resources.FindObjectsOfTypeAll<MageContextHud>().FirstOrDefault(h=>h.gameObject.scene==scene);
            if(hud==null)
            {
                var root=new GameObject("Mage-VL Context Test");
                if(!EditorApplication.isPlaying)Undo.RegisterCreatedObjectUndo(root,"Create Mage-VL Test");
                hud=root.AddComponent<MageContextHud>();
            }
            if(!EditorApplication.isPlaying)Undo.RecordObject(hud.gameObject,"Open Mage-VL Test");
            hud.gameObject.SetActive(true);Selection.activeGameObject=hud.gameObject;
            if(!EditorApplication.isPlaying)EditorSceneManager.MarkSceneDirty(scene);
            Debug.Log("SATORI_MAGE_VIEW_READY; enter Play Mode, edit Question Input, then click Ask Mage-VL.");
        }
    }
}
