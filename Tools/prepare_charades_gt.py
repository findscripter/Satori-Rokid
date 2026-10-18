"""Prepare a frozen 32-video CharadesEgo GT subset using bounded HTTP ranges.

Never extracts archive paths and never downloads the full archive. No model calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile
import threading
import time
import urllib.request
import uuid

URL="https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/CharadesEgo_v1_480.tar"
TOTAL=11375974400
SEED=20261016
BENCHMARK="charades-ego-32-v1"
SCENES=["Bedroom","Kitchen","Living room","Home Office / Study (A room in a house used for work)"]
MIB=1024*1024


def now():return datetime.now(timezone.utc).isoformat()
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(MIB),b""):h.update(block)
    return h.hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".writing")
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf-8");temp.replace(path)
def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))


def validate_range(status,headers,start,end,total=None,etag=None):
    headers={k.lower():v for k,v in headers.items()}
    match=re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",headers.get("content-range") or "")
    if status!=206 or not match:raise ValueError("Server did not return a valid 206 Content-Range")
    lo,hi,size=map(int,match.groups())
    if (lo,hi)!=(start,end) or (total is not None and size!=total):raise ValueError("Content-Range offset or archive size changed")
    returned=headers.get("etag")
    if not returned or (etag is not None and returned!=etag):raise ValueError("Archive ETag missing or changed")
    return size,returned


class RangeReader:
    def __init__(self,url=URL,total=TOTAL,etag=None,budget=500*MIB):
        self.url=url;self.total=total;self.etag=etag;self.budget=budget
        self.requested_bytes=0;self.received_bytes=0;self.requests=0;self.failures=[];self.lock=threading.Lock()
        self.get(0,511)
    def get(self,start,end):
        if start<0 or end<start or end>=self.total:raise ValueError("Range lies outside the archive")
        length=end-start+1
        with self.lock:
            if self.requested_bytes+length>self.budget:raise RuntimeError("Bounded media request budget exceeded")
            self.requested_bytes+=length;self.requests+=1
        headers={"User-Agent":"Satori-CharadesEgo-GT-preparation","Range":f"bytes={start}-{end}","Accept-Encoding":"identity"}
        if self.etag:headers["If-Match"]=self.etag
        try:
            request=urllib.request.Request(self.url,headers=headers)
            with urllib.request.urlopen(request,timeout=40) as response:
                total,etag=validate_range(response.status,dict(response.headers),start,end,self.total,self.etag)
                data=response.read(length+1)
            if len(data)!=length:raise ValueError("Range response body length mismatch")
            with self.lock:self.etag=etag;self.received_bytes+=len(data)
            return data
        except Exception as error:
            with self.lock:self.failures.append({"start":start,"end":end,"error_type":type(error).__name__,"error":str(error),"at_utc":now()})
            raise
    def report(self):
        return {"url":self.url,"total_bytes":self.total,"etag":self.etag,"requests":self.requests,"requested_bytes_upper_bound":self.requested_bytes,"received_body_bytes":self.received_bytes,"failures":list(self.failures)}


def parse_header(block,offset,total):
    if len(block)!=512 or offset%512 or block[257:263] not in (b"ustar\0",b"ustar "):return None
    try:item=tarfile.TarInfo.frombuf(block,"utf-8","strict")
    except (tarfile.HeaderError,ValueError,UnicodeError):return None
    next_header=offset+512+((item.size+511)//512)*512
    if item.size<0 or next_header>total:return None
    basename=PurePosixPath(item.name).name
    video_id=basename[:-4] if item.isfile() and basename.lower().endswith(".mp4") else None
    return {"name":item.name,"video_id":video_id,"header_offset":offset,"data_offset":offset+512,"size":item.size,"next_header":next_header,"header_sha256":hashlib.sha256(block).hexdigest()}


def find_header(reader,start):
    start=(start//512)*512;read_bytes=0
    for limit in (MIB,2*MIB,4*MIB):
        end=min(reader.total-1,start+limit-1)
        if start+read_bytes>end:break
        data=reader.get(start+read_bytes,end)
        for local in range(0,len(data)-511,512):
            header=parse_header(data[local:local+512],start+read_bytes+local,reader.total)
            if header is not None:return header
        read_bytes=end-start+1
    return None


def scan_lane(reader,lane,count):
    result={**lane};entries=[]
    if result.get("exhausted"):return result,entries
    try:
        if result.get("next_header") is None:
            header=find_header(reader,result["start"])
            if header is None:result.update(exhausted=True,failure="No valid aligned header within 4 MiB");return result,entries
        else:header=None
        for _ in range(count):
            if header is None:
                offset=result["next_header"]
                if offset+512>reader.total:result["exhausted"]=True;break
                header=parse_header(reader.get(offset,offset+511),offset,reader.total)
                if header is None:result.update(exhausted=True,failure="Next tar header failed checksum or was archive end");break
            if header["video_id"] is not None:entries.append(header)
            result["next_header"]=header["next_header"];result["headers_scanned"]=result.get("headers_scanned",0)+1;header=None
    except Exception as error:result.update(exhausted=True,failure=type(error).__name__+": "+str(error))
    return result,entries


def candidate_counts(rows,index):return Counter(row["scene"] for row in rows if row["id"] in index)


def select_samples(rows,index,scenes,per_scene,seed):
    selected=[]
    for scene in scenes:
        candidates=[row for row in rows if row["scene"]==scene and row["id"] in index]
        if len(candidates)<per_scene:raise ValueError(f"Insufficient indexed candidates for {scene}: {len(candidates)}")
        candidates.sort(key=lambda row:(hashlib.sha256((str(seed)+row["id"]).encode("utf-8")).hexdigest(),row["id"]))
        selected.extend(candidates[:per_scene])
    if len({row["id"] for row in selected})!=len(selected):raise ValueError("Duplicate selected source videos")
    return selected


def action_ids(value):
    ids={part.strip().split()[0] for part in (value or "").split(";") if part.strip()}
    if any(re.fullmatch(r"c\d{3}",item) is None for item in ids):raise ValueError("Unexpected official action ID")
    return sorted(ids)


def midpoint_indices(count,number=8):
    if count<number or number<=0:raise ValueError("Insufficient frames for distinct midpoint samples")
    return [math.floor((i+0.5)*count/number) for i in range(number)]


def frame_timestamp_ms(index,fps):
    if not math.isfinite(fps) or fps<=0:raise ValueError("Invalid decoder FPS")
    return round(index/fps*1000)


def load_catalog(data):
    source=data/"source";test=read_csv(source/"CharadesEgo_v1_test_only1st.csv")
    if len({row["id"] for row in test})!=len(test):raise ValueError("Duplicate official test IDs")
    scenes=sorted({row["scene"] for filename in ("CharadesEgo_v1_train.csv","CharadesEgo_v1_test.csv") for row in read_csv(source/filename) if row["scene"]})
    actions=[]
    for line in (source/"Charades_v1_classes.txt").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():identifier,label=line.strip().split(maxsplit=1);actions.append({"id":identifier,"label":label})
    actions.sort(key=lambda a:a["id"])
    if len(scenes)!=16 or [a["id"] for a in actions]!=[f"c{i:03d}" for i in range(157)]:raise ValueError("Official 16-scene/157-action taxonomy is incomplete")
    taxonomy={"schema_version":1,"scenes":[{"id":f"s{i:03d}","label":label} for i,label in enumerate(scenes)],"actions":actions}
    return test,taxonomy


def create_index_and_selection(args,reader,rows,taxonomy):
    data=args.data_root;selection_path=data/"selection.json"
    if selection_path.exists():
        selection=json.loads(selection_path.read_text(encoding="utf-8"))
        if selection["archive"]["etag"]!=reader.etag or selection["archive"]["total_bytes"]!=reader.total:raise ValueError("Frozen selection archive identity changed")
        return selection
    index_path=data/"media-index.json"
    if index_path.exists():
        old=json.loads(index_path.read_text(encoding="utf-8"))
        if old["archive"]["etag"]!=reader.etag:raise ValueError("Existing index ETag changed")
        index={entry["video_id"]:entry for entry in old["entries"]};lanes=old["lanes"]
    else:
        index={};lanes=[{"lane":i,"start":(reader.total*i//args.lanes)//512*512,"next_header":None,"headers_scanned":0} for i in range(args.lanes)]
    for round_no in range(args.max_index_rounds):
        counts=candidate_counts(rows,index)
        if all(counts[scene]>=8 for scene in SCENES):break
        with ThreadPoolExecutor(max_workers=args.workers) as executor:results=list(executor.map(lambda lane:scan_lane(reader,lane,args.headers_per_round),lanes))
        lanes=[]
        for lane,entries in results:
            lanes.append(lane)
            for entry in entries:
                existing=index.get(entry["video_id"])
                if existing is not None and existing["header_offset"]!=entry["header_offset"]:raise ValueError("Conflicting archive entries for the same video ID")
                index[entry["video_id"]]=entry
        counts=candidate_counts(rows,index)
        save(index_path,{"created_utc":now(),"archive":{"url":reader.url,"etag":reader.etag,"total_bytes":reader.total},"entries":sorted(index.values(),key=lambda e:e["header_offset"]),"lanes":lanes,"candidate_counts":dict(counts),"range_activity":reader.report()})
        print(json.dumps({"phase":"index","round":round_no+1,"headers":sum(l.get("headers_scanned",0) for l in lanes),"video_entries":len(index),"target_scene_counts":{s:counts[s] for s in SCENES},"bytes_read":reader.received_bytes}),flush=True)
    selected=select_samples(rows,index,SCENES,8,args.seed)
    candidates=[{"video_id":row["id"],"scene":row["scene"],"subject":row.get("subject"),"entry":index[row["id"]]} for row in rows if row["scene"] in SCENES and row["id"] in index]
    candidates.sort(key=lambda c:c["video_id"])
    selection={"schema_version":1,"benchmark_id":BENCHMARK,"frozen_utc":now(),"seed":args.seed,"sorting_rule":"ascending SHA256(UTF8(str(seed) + source_video_id)); source_video_id as tie-breaker","archive":{"url":reader.url,"etag":reader.etag,"total_bytes":reader.total},"candidate_pool":candidates,"media_index_sha256":sha256(index_path),"source_hashes":{filename:sha256(data/"source"/filename) for filename in ("CharadesEgo.zip","CharadesEgo_v1_test_only1st.csv","CharadesEgo_v1_train.csv","CharadesEgo_v1_test.csv","Charades_v1_classes.txt")},"samples":[{"sample_id":f"e{i:03d}","source_video_id":row["id"],"subject":row.get("subject"),"scene":row["scene"],"action_ids":action_ids(row["actions"]),"original_actions":row["actions"],"entry":index[row["id"]]} for i,row in enumerate(selected)],"protocol":{"split":"official test_only1st.csv","scene_strata":SCENES,"per_scene":8,"action_target":"deduplicated official whole-video action-ID set; anomalous original temporal intervals are retained, not corrected","frame_sampling":"8 uniform midpoint frame indices floor((i+0.5)*decoded_frame_count/8)","timestamp_basis":"decoded frame index divided by decoder-reported FPS; not fabricated 2 FPS","selection_before_model_inference":True,"availability_bias":"candidate pool is a deterministic sparse tar-index subset, not a full-archive random sample; failed/unscanned entries excluded before pool freeze only","replacement_policy":"frozen selected videos are never replaced after media/inference failure"}}
    if selection_path.exists():raise FileExistsError("Selection appeared concurrently")
    save(selection_path,selection);save(data/"taxonomy.json",taxonomy)
    print(json.dumps({"phase":"selection_frozen","candidates":len(candidates),"samples":len(selected),"selected_media_bytes":sum(item["entry"]["size"] for item in selection["samples"]),"selection_sha256":sha256(selection_path)}),flush=True)
    return selection


def download_video(args,reader,sample):
    directory=args.data_root/"media";directory.mkdir(parents=True,exist_ok=True)
    target=directory/(sample["sample_id"]+".mp4");entry=sample["entry"]
    record={"sample_id":sample["sample_id"],"source_video_id":sample["source_video_id"],"status":"downloading","url":reader.url,"etag":reader.etag,"archive_total_bytes":reader.total,"archive_member":entry["name"],"header_offset":entry["header_offset"],"data_start":entry["data_offset"],"data_end":entry["data_offset"]+entry["size"]-1,"expected_bytes":entry["size"],"path":target.relative_to(args.data_root).as_posix()}
    if target.exists():
        receipt=directory/(sample["sample_id"]+".json")
        if not receipt.exists():raise FileExistsError("Existing video has no receipt: "+target.name)
        prior=json.loads(receipt.read_text(encoding="utf-8"))
        if prior.get("status")!="completed" or prior.get("etag")!=reader.etag or prior.get("data_start")!=entry["data_offset"] or prior.get("sha256")!=sha256(target):raise ValueError("Existing media provenance does not verify")
        return prior
    part=directory/(sample["sample_id"]+"."+uuid.uuid4().hex+".part")
    try:
        with part.open("xb") as stream:
            offset=entry["data_offset"];end=offset+entry["size"]-1
            while offset<=end:
                upper=min(end,offset+4*MIB-1);stream.write(reader.get(offset,upper));offset=upper+1
        if part.stat().st_size!=entry["size"]:raise ValueError("Downloaded video size mismatch")
        record.update(status="completed",sha256=sha256(part),bytes=part.stat().st_size,finished_utc=now());part.replace(target)
    except Exception as error:record.update(status="failed",error_type=type(error).__name__,error=str(error),partial_path=part.relative_to(args.data_root).as_posix(),finished_utc=now())
    save(directory/(sample["sample_id"]+".json"),record)
    return record


def extract_frames(data,sample,media):
    import cv2
    video=data/media["path"];output=data/"frames"/sample["sample_id"]
    if output.exists():raise FileExistsError("Frame directory already exists; preserve it and inspect before retrying: "+str(output))
    output.mkdir(parents=True);capture=cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():raise ValueError("OpenCV cannot open selected video")
        count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(capture.get(cv2.CAP_PROP_FPS));frame_timestamp_ms(0,fps)
        indices=midpoint_indices(count);frames=[];details=[]
        for i,index in enumerate(indices):
            if not capture.set(cv2.CAP_PROP_POS_FRAMES,index):raise ValueError("Decoder seek failed")
            ok,image=capture.read()
            if not ok or image is None:raise ValueError("Decoder did not return the requested frame")
            reported=capture.get(cv2.CAP_PROP_POS_FRAMES)-1
            if abs(reported-index)>0.1:raise ValueError("Decoder frame position disagrees with requested index")
            frame_path=output/f"{i:03d}.jpg";ok,jpeg=cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,90])
            if not ok:raise ValueError("JPEG encoding failed")
            with frame_path.open("xb") as stream:stream.write(jpeg.tobytes())
            frames.append({"frame_id":f"v{i:03d}","timestamp_ms":frame_timestamp_ms(index,fps),"image_path":frame_path.relative_to(data).as_posix(),"sha256":sha256(frame_path)})
            details.append({"frame_id":f"v{i:03d}","source_frame_index_zero_based":index,"decoder_reported_position":reported,"source_timestamp_seconds":index/fps,"decoder_pos_msec":capture.get(cv2.CAP_PROP_POS_MSEC),"width":int(image.shape[1]),"height":int(image.shape[0]),"sha256":frames[-1]["sha256"]})
        return {"sample_id":sample["sample_id"],"frames":frames},{"sample_id":sample["sample_id"],"source_video_id":sample["source_video_id"],"video_sha256":media["sha256"],"decoded_frame_count":count,"decoder_fps":fps,"duration_from_count_fps_seconds":count/fps,"jpeg_quality":90,"frames":details}
    finally:capture.release()


def prepare(args):
    args.data_root=args.data_root.resolve();args.data_root.mkdir(parents=True,exist_ok=True)
    rows,taxonomy=load_catalog(args.data_root)
    reader=RangeReader(budget=args.max_download_mb*MIB)
    try:
        selection=create_index_and_selection(args,reader,rows,taxonomy)
        if args.stage=="index":return
        if (args.data_root/"infer-manifest.json").exists():raise FileExistsError("Prepared inference manifest already exists; refusing to replace the benchmark")
        with ThreadPoolExecutor(max_workers=min(args.workers,4)) as executor:
            media=list(executor.map(lambda sample:download_video(args,reader,sample),selection["samples"]))
        save(args.data_root/"media-provenance.json",{"archive":selection["archive"],"selection_sha256":sha256(args.data_root/"selection.json"),"videos":media,"range_activity":reader.report()})
        failures=[item for item in media if item["status"]!="completed"]
        if failures:raise RuntimeError(f"{len(failures)} frozen videos failed download; no sample replacements made")
        inference=[];frame_proofs=[]
        for sample,video in zip(selection["samples"],media):
            entry,proof=extract_frames(args.data_root,sample,video);inference.append(entry);frame_proofs.append(proof)
            print(json.dumps({"phase":"frames","sample_id":sample["sample_id"],"count":len(entry["frames"]),"fps":proof["decoder_fps"]}),flush=True)
        scene_ids={item["label"]:item["id"] for item in taxonomy["scenes"]};action_set={a["id"] for a in taxonomy["actions"]}
        for sample in selection["samples"]:
            if not set(sample["action_ids"]).issubset(action_set):raise ValueError("Official GT contains an unknown action ID")
        save(args.data_root/"frame-provenance.json",{"timestamp_basis":"source_frame_index / decoder_reported_fps; decoder_pos_msec also retained for audit","samples":frame_proofs})
        save(args.data_root/"taxonomy.json",taxonomy)
        save(args.data_root/"gt.json",{"schema_version":1,"benchmark_id":BENCHMARK,"labels":[{"sample_id":sample["sample_id"],"scene_id":scene_ids[sample["scene"]],"action_ids":sample["action_ids"],"source_video_id":sample["source_video_id"]} for sample in selection["samples"]],"provenance":{"source_hashes":selection["source_hashes"],"selection_sha256":sha256(args.data_root/"selection.json"),"media_provenance_sha256":sha256(args.data_root/"media-provenance.json"),"frame_provenance_sha256":sha256(args.data_root/"frame-provenance.json"),"labels":"official test_only1st.csv whole-video scene and deduplicated action set; original temporal intervals are not corrected"}})
        save(args.data_root/"infer-manifest.json",{"schema_version":1,"benchmark_id":BENCHMARK,"samples":inference})
        save(args.data_root/"preparation-result.json",{"status":"completed","prepared_utc":now(),"samples":len(inference),"frames":sum(len(i["frames"]) for i in inference),"source_video_count":len({s["source_video_id"] for s in selection["samples"]}),"range_activity":reader.report(),"model_calls":0,"infer_manifest_sha256":sha256(args.data_root/"infer-manifest.json"),"gt_sha256":sha256(args.data_root/"gt.json"),"taxonomy_sha256":sha256(args.data_root/"taxonomy.json")})
        print(json.dumps({"phase":"completed","samples":len(inference),"frames":256,"bytes_read":reader.received_bytes}),flush=True)
    finally:save(args.data_root/"range-activity.json",reader.report())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=Path("Research/Data/charades-ego-gt"));parser.add_argument("--stage",choices=("index","all"),default="all")
    parser.add_argument("--seed",type=int,default=SEED);parser.add_argument("--lanes",type=int,default=16);parser.add_argument("--headers-per-round",type=int,default=24)
    parser.add_argument("--max-index-rounds",type=int,default=8);parser.add_argument("--workers",type=int,default=8);parser.add_argument("--max-download-mb",type=int,default=500)
    args=parser.parse_args(argv)
    if args.seed!=SEED or min(args.lanes,args.headers_per_round,args.max_index_rounds,args.workers,args.max_download_mb)<=0:parser.error("Use frozen seed 20261016 and positive resource limits")
    try:prepare(args)
    except Exception as error:
        save(args.data_root/"preparation-failure.json",{"status":"failed","at_utc":now(),"error_type":type(error).__name__,"error":str(error),"model_calls":0,"selected_samples_replaced":False})
        print(json.dumps({"status":"failed","error_type":type(error).__name__,"error":str(error)},ensure_ascii=True),flush=True);return 1
    return 0


if __name__=="__main__":raise SystemExit(main())
