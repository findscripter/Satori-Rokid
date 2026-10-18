using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

namespace SatoriReplay.Editor
{
    public static class MageContextChecks
    {
        [Serializable] private sealed class ProofFrame
        {
            public string frame_id;public long timestamp_ms;public string sha256;public int texture_width,texture_height;
        }
        [Serializable] private sealed class Proof
        {
            public bool success;public string failure,request_state,request_id,question,run_directory,model,unity_version;
            public string request_file,response_file,response_sha256,raw_answer_sha256;
            public int passed,helper_exit_code,completed_request_count;public double latency_ms;
            public bool raw_answer_matches_ui,failure_guard_rejects_success,missing_imu,accepted_from_own_process;
            public string[] evidence_frame_ids,sensor_evidence_ids,missing_sensors,checks;
            public ProofFrame[] frames;
        }
        [MenuItem("Satori/Run Mage-VL Live Checks")]
        public static void RunLive()
        {
            var proof=new Proof{unity_version=Application.unityVersion};var checks=new List<string>();var frames=new List<ProofFrame>();
            var hud=UnityEngine.Object.FindObjectOfType<MageContextHud>();int original=hud!=null?hud.SelectedFrame:-1;
            Action<bool,string> check=(ok,name)=>{if(!ok)throw new InvalidOperationException(name);checks.Add(name);};
            try
            {
                check(EditorApplication.isPlaying,"Unity is in Play Mode");check(hud!=null,"Mage HUD is active");
                proof.request_state=hud.RequestState;proof.request_id=hud.RequestId;proof.completed_request_count=hud.CompletedRequestCount;
                check(hud.RequestState=="completed"&&hud.LastResult!=null,"Current request completed successfully");
                check(Guid.TryParse(hud.RequestId,out _),"Current request uses a GUID");
                var response=hud.LastResult;proof.question=response.question;proof.run_directory=response.run_directory;proof.model=response.model;proof.latency_ms=response.latency_ms;
                proof.request_file=hud.RequestFile;proof.response_file=hud.ResponseFile;proof.helper_exit_code=hud.HelperExitCode??-999;
                proof.accepted_from_own_process=hud.ResponseAcceptedFromOwnProcess;
                check(hud.ResponseAcceptedFromOwnProcess&&hud.HelperExitCode==0&&hud.CompletedRequestCount>0,"Response came from this client's successful helper process");
                check(File.Exists(hud.RequestFile)&&File.Exists(hud.ResponseFile)&&hud.RequestFile!=hud.ResponseFile,"Distinct request and response files exist");
                check(Path.GetFileName(hud.RequestFile)==hud.RequestId+".request.json"&&Path.GetFileName(hud.ResponseFile)==hud.RequestId+".response.json","Request files belong to this GUID");
                var request=MageJson.Read<MageRequest>(MageJson.ReadFile(hud.RequestFile));var saved=MageJson.Read<MageResponse>(MageJson.ReadFile(hud.ResponseFile));
                check(request.request_id==response.request_id&&saved.request_id==response.request_id&&response.request_id==hud.RequestId,"Request and response identities agree");
                check(saved.status=="completed"&&saved.question==request.question&&request.question==response.question&&response.question==hud.QuestionText.Trim(),"This result answers the current question");
                check(!string.IsNullOrWhiteSpace(response.run_directory)&&Directory.Exists(response.run_directory),"Real run directory exists");
                check(response.input_frames!=null&&response.input_frames.Length==8&&hud.InputFrames.Length==8&&request.window==8,"Eight real input frames are present");
                check(request.at_ms==9000&&response.input_frames[0].frame_id=="v011"&&response.input_frames[7].frame_id=="v018","Default causal window uses global anonymous frame IDs");
                long previous=-1;
                for(int i=0;i<8;i++)
                {
                    var frame=response.input_frames[i];var preview=hud.InputFrames[i];
                    check(frame.frame_id==preview.frame_id&&frame.timestamp_ms==preview.timestamp_ms&&frame.timestamp_ms>previous&&frame.timestamp_ms<=request.at_ms,"Frame identity and causal timestamp "+i);previous=frame.timestamp_ms;
                    check(File.Exists(frame.image_path)&&frame.sha256==MageContextHud.FileSha256(frame.image_path)&&frame.sha256==preview.sha256,"Real image SHA256 agrees with submitted input "+i);
                    hud.SelectFrame(i);var texture=hud.CurrentImage;
                    check(hud.SelectedFrameId==frame.frame_id&&texture!=null&&texture.width>0&&texture.height>0&&texture.name==frame.frame_id+"@"+frame.sha256,"Real image texture and selected frame agree "+i);
                    frames.Add(new ProofFrame{frame_id=frame.frame_id,timestamp_ms=frame.timestamp_ms,sha256=frame.sha256,texture_width=texture.width,texture_height=texture.height});
                }
                var answer=response.answer_payload;check(answer!=null,"Structured model answer is present");
                proof.raw_answer_matches_ui=hud.DisplayedRawAnswer==response.raw_answer_json&&response.raw_answer_json==saved.raw_answer_json;
                check(proof.raw_answer_matches_ui,"UI shows the complete raw JSON from this response without rewriting");
                check(MageJson.Write(MageJson.Read<MageAnswer>(response.raw_answer_json))==MageJson.Write(answer),"Raw answer JSON agrees with structured UI payload");
                check(hud.DisplayedDetails.Contains(answer.environment.place_type)&&hud.DisplayedDetails.Contains(answer.activity.action),"Environment and activity from the answer are visible");
                proof.evidence_frame_ids=answer.evidence_frame_ids;proof.sensor_evidence_ids=answer.sensor_evidence_ids;proof.missing_sensors=response.missing_sensors;
                var allowed=new HashSet<string>(response.input_frames.Select(f=>f.frame_id));
                check(answer.evidence_frame_ids!=null&&answer.evidence_frame_ids.Length>0&&answer.evidence_frame_ids.All(allowed.Contains),"Answer cites only actual input frames");
                proof.missing_imu=response.missing_sensors.Contains("imu")&&answer.sensor_evidence_ids!=null&&answer.sensor_evidence_ids.Length==0;
                check(proof.missing_imu&&hud.DisplayedDetails.Contains("无 IMU"),"No IMU input or fabricated sensor evidence");
                var texts=hud.GetComponentsInChildren<Text>();check(texts.Any(t=>t.name=="Provenance"&&t.text.Contains("非用户实拍")),"Public material is labelled as not user footage");
                check(hud.AskButton.interactable,"Ask button is available for another request");
                proof.failure_guard_rejects_success=!MageContextHud.CanPresentResult("failed",response,hud.RequestId)&&!MageContextHud.CanPresentResult("loading",response,hud.RequestId);
                check(proof.failure_guard_rejects_success,"Failed/loading state cannot authorize presentation of a previous successful answer");
                check(MageContextHud.CanPresentResult(hud.RequestState,response,hud.RequestId),"Current response is authorized for presentation");
                proof.response_sha256=MageContextHud.FileSha256(hud.ResponseFile);proof.raw_answer_sha256=HashText(response.raw_answer_json);
                proof.success=true;Debug.Log("SATORI_MAGE_LIVE_CHECKS_PASS "+checks.Count+" request="+hud.RequestId);
            }
            catch(Exception error){proof.failure=error.Message;Debug.LogError("SATORI_MAGE_LIVE_CHECKS_FAILED "+error.Message);}
            finally
            {
                if(hud!=null&&original>=0)hud.SelectFrame(original);proof.passed=checks.Count;proof.checks=checks.ToArray();proof.frames=frames.ToArray();Save("mage-live-tests.json",proof);
            }
            if(!proof.success)throw new InvalidOperationException(proof.failure);
        }
        [MenuItem("Satori/Check Mage-VL Failure State")]
        public static void CheckFailure()
        {
            var hud=UnityEngine.Object.FindObjectOfType<MageContextHud>();
            bool success=EditorApplication.isPlaying&&hud!=null&&hud.RequestState=="failed"&&hud.LastResult!=null&&hud.LastResult.status=="failed"&&
                hud.LastResult.answer_payload==null&&string.IsNullOrEmpty(hud.DisplayedRawAnswer)&&!hud.ResponseAcceptedFromOwnProcess&&hud.AskButton.interactable&&
                !MageContextHud.CanPresentResult(hud.RequestState,hud.LastResult,hud.RequestId);
            var proof=new Proof{success=success,request_state=hud?.RequestState,request_id=hud?.RequestId,run_directory=hud?.LastResult?.run_directory,unity_version=Application.unityVersion,
                passed=success?1:0,failure=success?null:"Failure UI still claims success, shows an old answer, or is not retryable.",failure_guard_rejects_success=success};
            Save("mage-failure-tests.json",proof);if(!success)throw new InvalidOperationException(proof.failure);
            Debug.Log("SATORI_MAGE_FAILURE_STATE_PASS request="+hud.RequestId);
        }
        private static string HashText(string text)
        {
            using(var hash=System.Security.Cryptography.SHA256.Create())return BitConverter.ToString(hash.ComputeHash(System.Text.Encoding.UTF8.GetBytes(text))).Replace("-","").ToLowerInvariant();
        }
        private static void Save(string name,Proof proof)
        {
            Directory.CreateDirectory("Assets/Satori/Evidence");File.WriteAllText("Assets/Satori/Evidence/"+name,JsonUtility.ToJson(proof,true));
        }
    }
}
