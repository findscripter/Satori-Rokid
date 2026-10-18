# Frozen source-dataset probes with VideoChat3-4B

This is a local-model adapter to `Tools/satori_paper_benchmark.py`. It is a
reconstructed module baseline on source datasets, not the unpublished original
Satori evaluation implementation. It does not load GT or source video names.

The owner must invoke `run.py` from a `gpurun -n 1` allocation. The inherited
backend loads only the existing pinned VideoChat3 snapshot onto logical cuda:0.
This directory does not launch a GPU job itself. Mage is not used.

The frozen prompt, question, candidates, cutoff and anonymous frame timestamps
are exactly the shared benchmark contract. Four-frame inputs remain four native
video frames; eight remain eight. Original FPS, source frame count and original
frame indices are provided as decoder technical metadata, with no resampling,
frame duplication or future image input. The processor's actual frame count is
checked after preprocessing.

Each sample receives one greedy generation with 4096 maximum output tokens.
There is no automatic retry, JSON extraction, repair or alternate answer parser.
A complete two-key JSON answer with actual frame evidence and EOS is required.
Raw generated text, EOS/length, usage, image/payload/prompt hashes, original
metadata and preprocessing are retained even when the prediction is invalid.
Local synchronous generation has no HTTP or hard wall-clock timeout; the trace
therefore records `timeout_seconds: null` explicitly.

CLI:

```text
python Tools/deploy/paper_videochat3/run.py --manifest PATH --taxonomy PATH --protocol PATH --metadata-registry PATH --model-path EXISTING_PINNED_SNAPSHOT --output NEW_RUN_DIR
```

The metadata registry has exactly this shape:

```json
{"benchmark_id":"same-as-manifest","samples":{"s000":{"fps":30.0,"total_num_frames":300,"duration":10.0,"width":720,"height":404,"frames_indices":[0,30,60,90],"video_backend":"opencv"}}}
```

Registry sample IDs must cover exactly the manifest; fields are restricted to
technical metadata. `duration` is seconds and must equal frame count / FPS.
Frame timestamps must agree with the frozen manifest to within 1 ms. Before
inference, the protocol must contain `technical_metadata_registry_sha256`, the
actual file byte SHA256. The registry bytes are saved as a run snapshot and
checked against the protocol, trace metadata and processor timeline. `audit.json`
is the shared benchmark audit; `metadata-audit.json` checks this extra contract.

CPU tests:

```text
Research/Backend/.venv/Scripts/python.exe -m pytest -q Tools/deploy/paper_videochat3/test_provider.py
```
