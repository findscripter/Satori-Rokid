"""Exercise both adapters on CPU and intercept before actual model generation."""
import argparse
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ("old-backend", "new-backend", "runtime", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(args.runtime))
    import torch
    from PIL import Image
    from transformers import AutoTokenizer, CLIPImageProcessor
    import inference
    import dispider.mm_utils as mm_utils
    from dispider.conversation import conv_templates

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=False)
    pd_tokenizer = AutoTokenizer.from_pretrained(args.model / "perception_decision", local_files_only=True)
    processor = CLIPImageProcessor.from_pretrained(args.model / "perception_decision/vision_tower", local_files_only=True)
    system = "Classify the scene and every visible action. Return only the required classification JSON."
    context = {"scene_candidates": [{"id": f"s{i:03d}", "label": f"scene {i}"} for i in range(16)],
               "action_candidates": [{"id": f"c{i:03d}", "label": f"action {i}"} for i in range(157)],
               "frames": [{"frame_id": f"v{i:03d}", "timestamp_ms": 1000 * (i + 1)} for i in range(8)]}
    images = [Image.new("RGB", (20, 16), (i * 20, 80, 160)) for i in range(8)]
    prepared_question = inference.preprocess_question
    reaction_tokenizer = mm_utils.tokenizer_image_token

    class InterceptedGeneration(Exception):
        pass

    def capture(path, tagged_images):
        spec = importlib.util.spec_from_file_location("checked_" + path.stem.replace("-", "_"), path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        records = {}

        class CapturedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(mm_use_im_start_end=tagged_images)
                self.decoder = torch.nn.Module()

            def get_model(self):
                return self.decoder

            def generate(self, input_ids, **kwargs):
                records["reaction_input_ids"] = input_ids.clone()
                records["generation_args"] = kwargs
                raise InterceptedGeneration

        backend = module.Backend.__new__(module.Backend)
        backend.torch, backend.device = torch, torch.device("cpu")
        backend.tokenizer, backend.time_tokenizer = tokenizer, pd_tokenizer
        backend.image_processor, backend.model = processor, CapturedModel()

        def capture_question(questions, time_tokenizer):
            records["pd_question"] = list(questions)
            return prepared_question(questions, time_tokenizer)

        def capture_prompt(prompt, *positional, **kwargs):
            records["reaction_prompt"] = prompt
            return reaction_tokenizer(prompt, *positional, **kwargs)

        inference.preprocess_question = capture_question
        mm_utils.tokenizer_image_token = capture_prompt
        try:
            backend.generate(system_prompt=system, context=context, frames=images,
                             max_new_tokens=4096, temperature=0, frame_metadata=None)
            raise AssertionError("Expected generation interception")
        except InterceptedGeneration:
            pass
        finally:
            inference.preprocess_question = prepared_question
            mm_utils.tokenizer_image_token = reaction_tokenizer
        return records

    checks = []
    for tagged in (False, True):
        old, new = capture(args.old_backend, tagged), capture(args.new_backend, tagged)
        original_user_data = old["pd_question"][0]
        assert new["pd_question"] == [system + "\n\n" + original_user_data]
        assert old["reaction_prompt"] == new["reaction_prompt"]
        assert torch.equal(old["reaction_input_ids"], new["reaction_input_ids"])
        assert old["reaction_prompt"].startswith(system)
        assert len(original_user_data.split("Frame ID:")) == 9
        assert system not in original_user_data
        for key in ("images", "images_large", "seqs", "compress_mask", "ans_token", "todo_token"):
            assert torch.equal(old["generation_args"][key], new["generation_args"][key]), key
        for key in ("max_new_tokens", "do_sample", "insert_position", "ans_position"):
            assert old["generation_args"][key] == new["generation_args"][key], key
        assert not torch.equal(old["generation_args"]["qs"], new["generation_args"]["qs"])
        checks.append({"mm_use_im_start_end": tagged, "status": "passed",
                       "pd_receives_original_system_plus_original_user_data": True,
                       "reaction_prompt_byte_identical": True, "reaction_input_token_ids_identical": True,
                       "images_time_sequences_delimiters_and_output_budget_identical": True,
                       "reaction_prompt_sha256": hashlib.sha256(new["reaction_prompt"].encode()).hexdigest(),
                       "old_pd_question_sha256": hashlib.sha256(original_user_data.encode()).hexdigest(),
                       "new_pd_question_sha256": hashlib.sha256(new["pd_question"][0].encode()).hexdigest()})
    old_text, new_text = args.old_backend.read_text(), args.new_backend.read_text()
    diff = "".join(difflib.unified_diff(old_text.splitlines(True), new_text.splitlines(True),
                                     fromfile="backend-before-pd-routing.py", tofile="backend.py"))
    expected_old = '        questions = preprocess_question([question], self.time_tokenizer)\n'
    expected_new = '        questions = preprocess_question([system_prompt + "\\n\\n" + question], self.time_tokenizer)\n'
    assert new_text == old_text.replace(expected_old, expected_new, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".patch").write_text(diff, encoding="utf-8")
    output = {"status": "passed", "device": "cpu", "checkpoint_weights_loaded": False,
              "actual_model_generation_executed": False, "ground_truth_read": False,
              "images": "eight synthetic solid-color images", "old_backend_sha256": digest(args.old_backend),
              "new_backend_sha256": digest(args.new_backend), "changed_source_lines": 1,
              "checks": checks, "native_streaming_changed": False,
              "scope": "Only PD instruction routing changed; silence and Reaction inputs remain intact."}
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
