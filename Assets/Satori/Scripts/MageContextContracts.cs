using System;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Text;

namespace SatoriReplay
{
    [DataContract] public sealed class MageRequest
    {
        [DataMember] public string request_id;
        [DataMember] public string question;
        [DataMember] public string session_path;
        [DataMember] public long at_ms;
        [DataMember] public int window;
    }
    [DataContract] public sealed class MageInputFrame
    {
        [DataMember] public string frame_id;
        [DataMember] public long timestamp_ms;
        [DataMember] public string image_path;
        [DataMember] public string sha256;
    }
    [DataContract] public sealed class MageEnvironment
    {
        [DataMember] public string place_type;
        [DataMember] public string[] visible_objects;
    }
    [DataContract] public sealed class MageActivity
    {
        [DataMember] public string action;
        [DataMember] public string phase;
    }
    [DataContract] public sealed class MageAnswer
    {
        [DataMember] public string answer;
        [DataMember] public MageEnvironment environment;
        [DataMember] public MageActivity activity;
        [DataMember] public string[] uncertainties;
        [DataMember] public string[] evidence_frame_ids;
        [DataMember] public string[] sensor_evidence_ids;
    }
    [DataContract] public sealed class MageUsage
    {
        [DataMember] public int? prompt_tokens;
        [DataMember] public int? completion_tokens;
        [DataMember] public int? total_tokens;
        [DataMember] public int? input_tokens;
        [DataMember] public int? output_tokens;
    }
    [DataContract] public sealed class MageResponse
    {
        [DataMember] public string status;
        [DataMember] public string request_id;
        [DataMember] public string question;
        [DataMember] public string run_directory;
        [DataMember] public string model;
        [DataMember] public MageAnswer answer_payload;
        [DataMember] public string raw_answer_json;
        [DataMember] public MageInputFrame[] input_frames;
        [DataMember] public double latency_ms;
        [DataMember] public MageUsage[] usage;
        [DataMember] public string[] missing_sensors;
        [DataMember] public string error_code;
        [DataMember] public string message;
    }
    [DataContract] public sealed class MagePreviewSession
    {
        [DataMember] public string session_id;
        [DataMember] public MageInputFrame[] frames;
    }
    public static class MageJson
    {
        public static T Read<T>(string value)
        {
            using(var stream=new MemoryStream(Encoding.UTF8.GetBytes(value)))
                return (T)new DataContractJsonSerializer(typeof(T)).ReadObject(stream);
        }
        public static string Write<T>(T value)
        {
            using(var stream=new MemoryStream())
            {
                new DataContractJsonSerializer(typeof(T)).WriteObject(stream,value);
                return Encoding.UTF8.GetString(stream.ToArray());
            }
        }
        public static string ReadFile(string path)
        {
            if(new FileInfo(path).Length>4*1024*1024)throw new IOException("JSON file exceeds 4 MB.");
            return File.ReadAllText(path,Encoding.UTF8);
        }
    }
}
