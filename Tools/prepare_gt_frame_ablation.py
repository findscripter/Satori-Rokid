"""Create a nested 8-to-32-frame CharadesEgo ablation without new media/model calls."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

BENCHMARK="charades-ego-32frames-v1"


def sha256(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):result.update(block)
    return result.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write("\n")


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def midpoints(count,number):
    if count<number or number<1:raise ValueError("Insufficient distinct source frames")
    return [math.floor((i+0.5)*count/number) for i in range(number)]


def nested_indices(frame_count,baseline_indices):
    if len(baseline_indices)!=8 or len(set(baseline_indices))!=8 or any(i<0 or i>=frame_count for i in baseline_indices):
        raise ValueError("Baseline must contain eight valid unique source-frame indices")
    if baseline_indices!=sorted(baseline_indices):raise ValueError("Baseline indices must be chronological")
    old=set(baseline_indices)
    remaining=[index for index in midpoints(frame_count,32) if index not in old]
    additions=[remaining[index] for index in midpoints(len(remaining),24)]
    selected=sorted(baseline_indices+additions)
    if len(selected)!=32 or len(set(selected))!=32 or not old.issubset(selected):raise ValueError("Nested frame selection failed")
    return selected


def copy_baseline_frame(source,target,expected_hash):
    raw=Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_hash:raise ValueError("Original eight-frame JPEG hash no longer matches")
    with Path(target).open("xb") as stream:stream.write(raw)
    if sha256(target)!=expected_hash:raise ValueError("Copied JPEG is not byte-identical")


def ablation_gt(original,source_provenance):
    return {"schema_version":original["schema_version"],"benchmark_id":BENCHMARK,"labels":copy.deepcopy(original["labels"]),
            "provenance":{"source_gt":copy.deepcopy(source_provenance),"original_provenance":copy.deepcopy(original.get("provenance",{})),"label_policy":"Exact same sample IDs, source video IDs, scene IDs and deduplicated whole-video action IDs as the frozen 8-frame benchmark; no correction, reselection or relabelling."}}


def contained(root,relative):
    candidate=(root/relative).resolve()
    if not candidate.is_relative_to(root):raise ValueError("Source artifact path escapes the frozen data directory")
    return candidate


def prepare_sample(source,target,sample,old_proof,video):
    import cv2
    sample_id=sample["sample_id"]
    if not sample_id.startswith("e") or not sample_id[1:].isdigit():raise ValueError("Unsafe sample ID")
    video_path=contained(source,video["path"])
    if sha256(video_path)!=video["sha256"] or video["sha256"]!=old_proof["video_sha256"]:raise ValueError("Source video hash changed")
    old_by_id={frame["frame_id"]:frame for frame in sample["frames"]}
    if len(old_by_id)!=8 or len(old_proof["frames"])!=8:raise ValueError("Baseline frame count is not eight")
    old_by_index={}
    for proof in old_proof["frames"]:
        frame=old_by_id[proof["frame_id"]]
        if frame["sha256"]!=proof["sha256"]:raise ValueError("Baseline manifest/provenance hashes disagree")
        old_by_index[proof["source_frame_index_zero_based"]]=(frame,proof)
    indices=nested_indices(old_proof["decoded_frame_count"],sorted(old_by_index))
    folder=target/"frames"/sample_id;folder.mkdir(parents=True,exist_ok=False)
    capture=cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():raise ValueError("Cannot open source video")
        count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(capture.get(cv2.CAP_PROP_FPS))
        if count!=old_proof["decoded_frame_count"] or not math.isfinite(fps) or fps<=0 or abs(fps-old_proof["decoder_fps"])>1e-8:raise ValueError("Decoder frame count/FPS changed from baseline")
        frames=[];proofs=[];copied=0
        for ordinal,index in enumerate(indices):
            path=folder/f"{ordinal:03d}.jpg";timestamp=round(index/fps*1000)
            if index in old_by_index:
                original,old_frame_proof=old_by_index[index]
                old_path=contained(source,original["image_path"])
                if timestamp!=original["timestamp_ms"]:raise ValueError("Preserved baseline timestamp changed")
                copy_baseline_frame(old_path,path,original["sha256"]);copied+=1
                proof={**copy.deepcopy(old_frame_proof),"origin":"byte_identical_baseline_copy","original_frame_id":original["frame_id"],"original_image_path":str(old_path),"original_sha256":original["sha256"]}
            else:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES,index):raise ValueError("Source frame seek failed")
                ok,image=capture.read()
                if not ok or image is None:raise ValueError("Failed to decode an additional frame")
                position=capture.get(cv2.CAP_PROP_POS_FRAMES)-1
                if abs(position-index)>0.1:raise ValueError("Decoder did not return the requested source frame")
                ok,jpeg=cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,old_proof.get("jpeg_quality",90)])
                if not ok:raise ValueError("JPEG encoding failed")
                with path.open("xb") as stream:stream.write(jpeg.tobytes())
                proof={"origin":"additional_32_midpoint_candidate","source_frame_index_zero_based":index,"decoder_reported_position":position,"source_timestamp_seconds":index/fps,"decoder_pos_msec":capture.get(cv2.CAP_PROP_POS_MSEC),"width":int(image.shape[1]),"height":int(image.shape[0])}
            digest=sha256(path);frame_id=f"v{ordinal:03d}"
            proof.update(frame_id=frame_id,sha256=digest,timestamp_ms=timestamp)
            proofs.append(proof);frames.append({"frame_id":frame_id,"timestamp_ms":timestamp,"image_path":path.relative_to(target).as_posix(),"sha256":digest})
        if copied!=8 or any(a["timestamp_ms"]>=b["timestamp_ms"] for a,b in zip(frames,frames[1:])):raise ValueError("Frame nesting or timestamp ordering failed")
        source_video={**copy.deepcopy(video),"original_media_path":str(video_path),"reused_without_download":True}
        proof={"sample_id":sample_id,"source_video_id":old_proof["source_video_id"],"video_sha256":video["sha256"],"decoded_frame_count":count,"decoder_fps":fps,"jpeg_quality":old_proof.get("jpeg_quality",90),"baseline_source_indices":sorted(old_by_index),"midpoint32_candidate_indices":midpoints(count,32),"selected_source_indices":indices,"copied_baseline_frames":copied,"additional_frames":24,"frames":proofs}
        save(target/"sample-provenance"/(sample_id+".json"),proof)
        return {"sample_id":sample_id,"frames":frames},proof,source_video
    finally:capture.release()


def prepare(source,target,workers=4):
    import cv2
    cv2.setNumThreads(1)
    source=source.resolve(strict=True);target=target.resolve()
    if target==source or target.is_relative_to(source):raise ValueError("Ablation must use a new independent data directory")
    if target.exists():raise FileExistsError("Output directory already exists; never overwrite a prepared ablation")
    names=("taxonomy.json","gt.json","infer-manifest.json","frame-provenance.json","media-provenance.json","selection.json")
    hashes={name:sha256(source/name) for name in names}
    manifest=read(source/"infer-manifest.json");old_gt=read(source/"gt.json");old_frames=read(source/"frame-provenance.json");old_media=read(source/"media-provenance.json")
    samples=manifest["samples"];sample_ids=[sample["sample_id"] for sample in samples]
    if len(samples)!=32 or len(set(sample_ids))!=32:raise ValueError("Expected exactly 32 distinct frozen samples")
    labels={label["sample_id"]:label for label in old_gt["labels"]};proofs={p["sample_id"]:p for p in old_frames["samples"]};media={v["sample_id"]:v for v in old_media["videos"]}
    if set(sample_ids)!=set(labels) or set(sample_ids)!=set(proofs) or set(sample_ids)!=set(media):raise ValueError("Frozen source files have different sample sets")
    if len({label["source_video_id"] for label in old_gt["labels"]})!=32:raise ValueError("Source videos are not distinct")
    for sample_id in sample_ids:
        if labels[sample_id]["source_video_id"]!=proofs[sample_id]["source_video_id"] or labels[sample_id]["source_video_id"]!=media[sample_id]["source_video_id"]:raise ValueError("Source video mappings disagree")
    target.mkdir(parents=True,exist_ok=False)
    protocol={"schema_version":1,"benchmark_id":BENCHMARK,"frozen_utc":datetime.now(timezone.utc).isoformat(),"source_root":str(source),"source_benchmark_id":manifest["benchmark_id"],"source_file_sha256":hashes,"sample_count":32,"frames_per_sample":32,"baseline_frames_preserved_per_sample":8,"new_frames_per_sample":24,"selection_rule":"Keep the baseline eight source indices and byte-identical JPEGs. Form all 32 uniform midpoint source-frame candidates; remove baseline indices; take 24 candidates at floor((j+0.5)*remaining_count/24), then sort all 32 by source index.","not_plain_midpoint32":True,"controlled_factors":["same 32 source videos and official whole-video GT","same scene/action taxonomy","same JPEG quality for new frames","all original eight JPEGs copied byte-for-byte"],"changed_factors":["additional temporal coverage and visual token/context load; result does not isolate coverage from token budget"],"timestamp_basis":"zero-based source frame index / decoder-reported FPS; baseline timestamps remain exactly unchanged","new_media_downloads":0,"model_calls":0}
    save(target/"ablation-protocol.json",protocol)
    try:
        inference=[];frame_proofs=[];video_proofs=[]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            jobs=executor.map(lambda sample:prepare_sample(source,target,sample,proofs[sample["sample_id"]],media[sample["sample_id"]]),samples)
            for entry,frame_proof,video_proof in jobs:
                inference.append(entry);frame_proofs.append(frame_proof);video_proofs.append(video_proof)
                print(json.dumps({"phase":"prepared","sample_id":entry["sample_id"],"frames":32,"copied_baseline_frames":8}),flush=True)
        if any(sha256(source/name)!=digest for name,digest in hashes.items()):raise ValueError("Frozen source metadata changed during preparation")
        # Taxonomy is also a byte-identical copy; the GT labels are deep-copied unchanged.
        with (target/"taxonomy.json").open("xb") as stream:stream.write((source/"taxonomy.json").read_bytes())
        save(target/"source-video-provenance.json",{"source_media_provenance_path":str(source/"media-provenance.json"),"source_media_provenance_sha256":hashes["media-provenance.json"],"archive":old_media["archive"],"videos":video_proofs})
        save(target/"frame-provenance.json",{"protocol_sha256":sha256(target/"ablation-protocol.json"),"timestamp_basis":protocol["timestamp_basis"],"samples":frame_proofs})
        save(target/"gt.json",ablation_gt(old_gt,{"path":str(source/"gt.json"),"sha256":hashes["gt.json"],"selection_sha256":hashes["selection.json"]}))
        save(target/"infer-manifest.json",{"schema_version":1,"benchmark_id":BENCHMARK,"samples":inference})
        save(target/"preparation-result.json",{"status":"completed","prepared_utc":datetime.now(timezone.utc).isoformat(),"samples":32,"frames":1024,"byte_identical_baseline_frames":256,"newly_decoded_frames":768,"new_media_downloads":0,"model_calls":0,"original_metadata_hashes_unchanged":True,"original_gt_labels_unchanged":read(target/"gt.json")["labels"]==old_gt["labels"],"taxonomy_byte_identical":sha256(target/"taxonomy.json")==hashes["taxonomy.json"],"manifest_sha256":sha256(target/"infer-manifest.json"),"gt_sha256":sha256(target/"gt.json"),"script_sha256":sha256(Path(__file__))})
        print(json.dumps({"status":"completed","samples":32,"frames":1024,"byte_identical_baseline_frames":256,"output":str(target)}),flush=True)
    except Exception as error:
        save(target/"preparation-failure.json",{"status":"failed","error_type":type(error).__name__,"error":str(error),"model_calls":0,"old_data_modified":False})
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,default=Path("Research/Data/charades-ego-gt"))
    parser.add_argument("--output-root",type=Path,default=Path("Research/Data/charades-ego-32frames-v1"))
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args(argv)
    if not 1<=args.workers<=8:parser.error("workers must be between 1 and 8")
    prepare(args.source_root,args.output_root,args.workers)


if __name__=="__main__":main()
