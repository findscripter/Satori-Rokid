using System;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

namespace SatoriReplay.Editor
{
    public static class EvidenceViewSetup
    {
        private const int Width=1440,Height=940;
        private const BindingFlags InstanceFlags=BindingFlags.Instance|BindingFlags.Public|BindingFlags.NonPublic;

        [MenuItem("Satori/Prepare Evidence View")]
        public static void Prepare()
        {
            var assembly=typeof(EditorWindow).Assembly;
            var sizesType=assembly.GetType("UnityEditor.GameViewSizes",true);
            var sizeType=assembly.GetType("UnityEditor.GameViewSize",true);
            var sizeModeType=assembly.GetType("UnityEditor.GameViewSizeType",true);
            var groupType=assembly.GetType("UnityEditor.GameViewSizeGroupType",true);
            var singleton=typeof(ScriptableSingleton<>).MakeGenericType(sizesType);
            var instanceProperty=singleton.GetProperty("instance",BindingFlags.Public|BindingFlags.Static);
            if(instanceProperty==null)throw new MissingMemberException("Unity GameViewSizes.instance is unavailable.");
            var sizes=instanceProperty.GetValue(null,null);
            var currentGroup=sizesType.GetProperty("currentGroupType",InstanceFlags)?.GetValue(sizes,null)??Enum.Parse(groupType,"Standalone");
            var group=sizesType.GetMethod("GetGroup",InstanceFlags)?.Invoke(sizes,new[]{currentGroup});
            if(group==null)throw new MissingMethodException("Unity GameViewSizes.GetGroup is unavailable.");
            var groupClass=group.GetType();
            var countMethod=groupClass.GetMethod("GetTotalCount",InstanceFlags);
            int count=countMethod!=null?(int)countMethod.Invoke(group,null):
                (int)groupClass.GetMethod("GetBuiltinCount",InstanceFlags).Invoke(group,null)+(int)groupClass.GetMethod("GetCustomCount",InstanceFlags).Invoke(group,null);
            var getSize=groupClass.GetMethod("GetGameViewSize",InstanceFlags);
            var width=sizeType.GetProperty("width",InstanceFlags);var height=sizeType.GetProperty("height",InstanceFlags);
            var mode=sizeType.GetProperty("sizeType",InstanceFlags);
            if(getSize==null||width==null||height==null||mode==null)throw new MissingMemberException("Unity GameView size reflection API changed.");
            var fixedResolution=Enum.Parse(sizeModeType,"FixedResolution");int selected=-1;
            for(int i=0;i<count;i++)
            {
                var size=getSize.Invoke(group,new object[]{i});
                if((int)width.GetValue(size,null)==Width&&(int)height.GetValue(size,null)==Height&&Equals(mode.GetValue(size,null),fixedResolution)){selected=i;break;}
            }
            if(selected<0)
            {
                var constructor=sizeType.GetConstructor(InstanceFlags,null,new[]{sizeModeType,typeof(int),typeof(int),typeof(string)},null);
                if(constructor==null)throw new MissingMethodException("Unity GameViewSize constructor is unavailable.");
                var size=constructor.Invoke(new object[]{fixedResolution,Width,Height,"Satori Evidence 1440 x 940"});
                var add=groupClass.GetMethod("AddCustomSize",InstanceFlags);
                if(add==null)throw new MissingMethodException("Unity GameViewSizeGroup.AddCustomSize is unavailable.");
                add.Invoke(group,new[]{size});selected=count;
            }
            var viewType=assembly.GetType("UnityEditor.GameView",true);
            var window=EditorWindow.GetWindow(viewType);
            var callback=viewType.GetMethod("SizeSelectionCallback",InstanceFlags,null,new[]{typeof(int),typeof(object)},null);
            if(callback!=null)callback.Invoke(window,new object[]{selected,null});
            else
            {
                var selection=viewType.GetProperty("selectedSizeIndex",InstanceFlags);
                if(selection==null||!selection.CanWrite)throw new MissingMemberException("Unity GameView size selection is unavailable.");
                selection.SetValue(window,selected,null);
            }
            var lowRes=viewType.GetProperty("lowResolutionForAspectRatios",InstanceFlags);
            if(lowRes!=null&&lowRes.CanWrite)lowRes.SetValue(window,false,null);
            window.Show();window.Focus();window.Repaint();
            EditorApplication.QueuePlayerLoopUpdate();
            Debug.Log("SATORI_EVIDENCE_VIEW_REQUESTED 1440x940 fixed resolution; wait for repaint before capture.");
        }

        [MenuItem("Satori/Check Evidence Image Bounds")]
        public static void CheckBounds()
        {
            var hud=UnityEngine.Object.FindObjectOfType<ReplayHud>();
            if(!EditorApplication.isPlaying||hud==null||!hud.HasLoadedImage)throw new InvalidOperationException("A real image must be loaded in Play Mode before checking evidence bounds.");
            Canvas.ForceUpdateCanvases();
            var raw=hud.GetComponentInChildren<RawImage>();
            var backdrop=raw.transform.parent.Find("Image Backdrop") as RectTransform;
            if(backdrop==null)throw new InvalidOperationException("Evidence image backdrop was not found.");
            var corners=new Vector3[4];raw.rectTransform.GetWorldCorners(corners);
            const float tolerance=0.1f;
            foreach(var corner in corners)
            {
                var local=backdrop.InverseTransformPoint(corner);var bounds=backdrop.rect;
                if(local.x<bounds.xMin-tolerance||local.x>bounds.xMax+tolerance||local.y<bounds.yMin-tolerance||local.y>bounds.yMax+tolerance)
                    throw new InvalidOperationException("The displayed image extends beyond its evidence viewport.");
            }
            var rect=raw.rectTransform.rect;
            float inputRatio=(float)raw.texture.width/raw.texture.height,displayRatio=rect.width/rect.height;
            if(Mathf.Abs(inputRatio-displayRatio)>0.001f)throw new InvalidOperationException("Displayed image aspect ratio is distorted.");
            Directory.CreateDirectory("Assets/Satori/Evidence");
            File.WriteAllText("Assets/Satori/Evidence/image-bounds.json",JsonUtility.ToJson(new BoundsProof{success=true,current_frame=hud.CurrentFrame,display_width=rect.width,display_height=rect.height,viewport_width=backdrop.rect.width,viewport_height=backdrop.rect.height,screen_width=Screen.width,screen_height=Screen.height},true));
            Debug.Log("SATORI_IMAGE_BOUNDS_PASS contained and aspect-preserving: "+rect.width+"x"+rect.height);
        }
        [Serializable] private sealed class BoundsProof
        {
            public bool success;public int current_frame;public float display_width,display_height,viewport_width,viewport_height;public int screen_width,screen_height;
        }

        [MenuItem("Satori/Check Timeline Bounds")]
        public static void CheckTimeline()
        {
            var hud=UnityEngine.Object.FindObjectOfType<ReplayHud>();
            if(!EditorApplication.isPlaying||hud==null||hud.LoadedReport==null||hud.LoadedReport.records.Length<2)
                throw new InvalidOperationException("Load at least two replay records in Play Mode before checking the timeline.");
            var slider=hud.GetComponentInChildren<Slider>();
            if(slider==null||slider.fillRect==null||slider.handleRect==null)throw new InvalidOperationException("Timeline fill or handle is missing.");
            var track=slider.transform as RectTransform;
            int original=hud.CurrentFrame,last=hud.LoadedReport.records.Length-1;
            bool wasPlaying=hud.IsPlaying;
            var proof=new TimelineProof{original_frame=original,cases=new TimelineCase[3]};
            var indices=new[]{0,hud.LoadedReport.records.Length/2,last};
            try
            {
                for(int i=0;i<indices.Length;i++)
                {
                    hud.SeekFrame(indices[i]);Canvas.ForceUpdateCanvases();
                    float fraction=(float)indices[i]/last,expected=track.rect.width*fraction;
                    float actual=slider.fillRect.rect.width;
                    var handleCenter=track.InverseTransformPoint(slider.handleRect.TransformPoint(slider.handleRect.rect.center));
                    var sample=new TimelineCase{frame=indices[i],fraction=fraction,fill_width=actual,expected_width=expected,track_width=track.rect.width,handle_center_x=handleCenter.x,handle_width=slider.handleRect.rect.width,handle_height=slider.handleRect.rect.height};
                    proof.cases[i]=sample;
                    if(Mathf.Abs(actual-expected)>0.1f)throw new InvalidOperationException("Timeline fill width is incorrect at frame "+indices[i]+": expected "+expected+", got "+actual);
                    if(Mathf.Abs(handleCenter.x-(track.rect.xMin+expected))>0.1f)throw new InvalidOperationException("Timeline handle does not match the selected frame.");
                    if(Mathf.Abs(sample.handle_width-24)>0.1f||Mathf.Abs(sample.handle_height-27)>0.1f)throw new InvalidOperationException("Timeline handle geometry changed with its anchors.");
                }
                proof.success=true;
                Debug.Log("SATORI_TIMELINE_BOUNDS_PASS first/middle/last fill and handle positions");
            }
            catch(Exception error){proof.failure=error.Message;throw;}
            finally
            {
                hud.SeekFrame(original);if(wasPlaying)hud.TogglePlayback();
                Directory.CreateDirectory("Assets/Satori/Evidence");
                File.WriteAllText("Assets/Satori/Evidence/timeline-bounds.json",JsonUtility.ToJson(proof,true));
            }
        }
        [Serializable] private sealed class TimelineProof
        {
            public bool success;public string failure;public int original_frame;public TimelineCase[] cases;
        }
        [Serializable] private sealed class TimelineCase
        {
            public int frame;public float fraction,fill_width,expected_width,track_width,handle_center_x,handle_width,handle_height;
        }
    }
}
