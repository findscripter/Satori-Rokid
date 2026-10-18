using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Text;

namespace SatoriReplay
{
    [DataContract] public sealed class ReplayReport
    {
        [DataMember(IsRequired=true)] public string session_id;
        [DataMember(IsRequired=true)] public bool synthetic;
        [DataMember] public string source_kind;
        [DataMember] public string source_label;
        [DataMember(IsRequired=true)] public ReplayRecord[] records;
    }
    [DataContract] public sealed class ReplayRecord
    {
        [DataMember(IsRequired=true)] public string frame_id;
        [DataMember(IsRequired=true)] public long timestamp_ms;
        [DataMember] public string scene;
        [DataMember] public string action;
        [DataMember] public string current_step_id;
        [DataMember] public string current_step_instruction;
        [DataMember] public string task_title;
        [DataMember(IsRequired=true)] public double confidence;
        [DataMember] public bool uncertain;
        [DataMember] public string[] confirmed_completed_step_ids;
        [DataMember] public string[] observed_completed_step_ids;
        [DataMember(IsRequired=true)] public string decision_kind;
        [DataMember] public string decision_message;
        [DataMember] public string[] evidence_frame_ids;
        [DataMember] public string image_path;
    }
    public static class ReportParser
    {
        public static ReplayReport Load(string path)
        {
            if(new FileInfo(path).Length>32*1024*1024) throw new ArgumentException("报告超过 32 MB。");
            return Parse(File.ReadAllText(path,Encoding.UTF8));
        }
        public static ReplayReport Parse(string json)
        {
            ReplayReport report;
            try { using(var input=new MemoryStream(Encoding.UTF8.GetBytes(json??""))) report=(ReplayReport)new DataContractJsonSerializer(typeof(ReplayReport)).ReadObject(input); }
            catch(Exception e) when(e is SerializationException || e is System.Xml.XmlException || e is ArgumentException)
            { throw new ArgumentException("报告 JSON 无效或缺少必填字段。",e); }
            if(report==null || string.IsNullOrWhiteSpace(report.session_id) || report.records==null || report.records.Length==0)
                throw new ArgumentException("必须提供 session_id、synthetic 和非空 records。");
            var frames=new HashSet<string>(StringComparer.Ordinal); long previous=-1;
            foreach(var item in report.records)
            {
                if(item==null || string.IsNullOrWhiteSpace(item.frame_id) || !frames.Add(item.frame_id)) throw new ArgumentException("帧 ID 必须非空且唯一。");
                if(item.timestamp_ms<0 || item.timestamp_ms<=previous) throw new ArgumentException("时间戳必须非负且严格递增。");
                previous=item.timestamp_ms;
                if(double.IsNaN(item.confidence)||double.IsInfinity(item.confidence)||item.confidence<0||item.confidence>1) throw new ArgumentException("confidence 必须是 0 到 1 的有限数字。");
                if(item.decision_kind!="observe"&&item.decision_kind!="clarify"&&item.decision_kind!="guide"&&item.decision_kind!="complete") throw new ArgumentException("未知辅助决策类型。");
                item.confirmed_completed_step_ids=Normalize(item.confirmed_completed_step_ids);
                item.observed_completed_step_ids=Normalize(item.observed_completed_step_ids);
                item.evidence_frame_ids=Normalize(item.evidence_frame_ids);
                foreach(var id in item.evidence_frame_ids) if(!frames.Contains(id)) throw new ArgumentException("证据引用未知或未来帧："+id);
            }
            return report;
        }
        private static string[] Normalize(string[] values)
        {
            if(values==null)return Array.Empty<string>();
            var seen=new HashSet<string>(StringComparer.Ordinal);
            foreach(var value in values) if(string.IsNullOrWhiteSpace(value)||!seen.Add(value)) throw new ArgumentException("步骤或证据 ID 必须非空且唯一。");
            return values;
        }
    }
    public sealed class ReplayPlayback
    {
        public ReplayReport Report{get;}
        public int Index{get;private set;}
        public bool Playing{get;private set;}
        public ReplayRecord Current=>Report.records[Index];
        private double clockMs,speed=1;
        public double Speed{get=>speed;set{if(double.IsNaN(value)||double.IsInfinity(value)||value<=0||value>16)throw new ArgumentOutOfRangeException(nameof(value));speed=value;}}
        public ReplayPlayback(ReplayReport report)
        {
            if(report?.records==null||report.records.Length==0)throw new ArgumentException("空时间轴。");
            Report=report;clockMs=Current.timestamp_ms;
        }
        public void Seek(int index){Index=Math.Max(0,Math.Min(index,Report.records.Length-1));clockMs=Current.timestamp_ms;Playing=false;}
        public void Previous()=>Seek(Index-1);
        public void Next()=>Seek(Index+1);
        public void Toggle(){if(Report.records.Length==1)return;if(Index==Report.records.Length-1)Seek(0);Playing=!Playing;}
        public void Advance(double seconds)
        {
            if(double.IsNaN(seconds)||double.IsInfinity(seconds)||seconds<0)throw new ArgumentOutOfRangeException(nameof(seconds));
            if(!Playing)return;clockMs+=seconds*1000*Speed;
            while(Index+1<Report.records.Length&&Report.records[Index+1].timestamp_ms<=clockMs+0.0001)Index++;
            if(Index==Report.records.Length-1){clockMs=Current.timestamp_ms;Playing=false;}
        }
    }
}
