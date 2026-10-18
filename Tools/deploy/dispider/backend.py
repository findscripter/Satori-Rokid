"""Dispider's offline Reaction path for the frozen eight-image comparison.

This retains Perception/Decision memory construction but does not exercise the
native streaming Decision trigger. Eight unique images occupy 16 slots by
adjacent duplication. The checkpoint is unchanged; SDPA is a recorded runtime
compatibility adaptation.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def compact_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


class Backend:
    model_id = "Dispider"

    def __init__(self, model_path: str):
        if not os.environ.get("GPURUN_JOB") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise RuntimeError("GPU inference must be launched through gpurun")
        runtime = Path(os.environ["DISPIDER_RUNTIME"]).resolve()
        sys.path.insert(0, str(runtime))
        os.environ["DISPIDER_ATTN_IMPLEMENTATION"] = "sdpa"
        import torch
        from dispider.model.builder import load_pretrained_model

        self.torch = torch
        self.tokenizer, self.model, processors, self.context_len = load_pretrained_model(
            model_path, device_map={"": "cuda:0"}, attn_implementation="sdpa",
            local_files_only=True)
        self.model.eval()
        self.image_processor, self.time_tokenizer = processors
        self.device = torch.device("cuda:0")
        if self.time_tokenizer.pad_token is None:
            self.time_tokenizer.pad_token = "<pad>"

    def generate(self, *, system_prompt: str, context: dict, frames: list,
                 max_new_tokens: int, temperature: float = 0,
                 frame_metadata: dict | None = None) -> dict:
        if len(frames) != 8 or len(context.get("frames", [])) != 8:
            raise ValueError("Frozen comparison requires exactly eight unique images")
        if temperature != 0:
            raise ValueError("This comparison requires deterministic greedy generation")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 8192:
            raise ValueError("Invalid output budget")
        import numpy as np
        from PIL import Image
        from transformers import StoppingCriteriaList
        from inference import StoppingCriteriaSub, preprocess_time, preprocess_question
        from dispider.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
                                       DEFAULT_IM_END_TOKEN, DEFAULT_ANS_TOKEN,
                                       DEFAULT_TODO_TOKEN, DEFAULT_SILENT_TOKEN,
                                       IMAGE_TOKEN_INDEX)
        from dispider.conversation import conv_templates
        from dispider.mm_utils import tokenizer_image_token

        torch = self.torch
        frame_labels = [f"Frame ID: {entry['frame_id']}; timestamp_ms: {entry['timestamp_ms']}"
                        for entry in context["frames"]]
        question = compact_json(context) + "\n" + "\n".join(frame_labels)
        user_text = DEFAULT_IMAGE_TOKEN + "\n" + question
        if getattr(self.model.config, "mm_use_im_start_end", False):
            user_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
        conversation = conv_templates["qwen"].copy()
        conversation.system = system_prompt
        conversation.append_message(conversation.roles[0], user_text)
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()

        unique = [np.asarray(frame.convert("RGB")) for frame in frames]
        if len({value.shape for value in unique}) != 1:
            raise ValueError("All frozen images must have the same shape")
        array = np.stack([value for value in unique for _ in range(2)])
        _, height, width, _ = array.shape
        # Same nearest-neighbor square resize as upstream inference.load_video.
        if height != width:
            square_size = min(height, width)
            tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float()
            tensor = torch.nn.functional.interpolate(tensor, size=(square_size, square_size))
            array = tensor.permute(0, 2, 3, 1).to(torch.uint8).numpy()
        slots = [Image.fromarray(value) for value in array]
        video = self.image_processor.preprocess(slots, return_tensors="pt")["pixel_values"]
        video = video.view(1, 16, *video.shape[1:])

        stamps = [entry["timestamp_ms"] / 1000 for entry in context["frames"]]
        time_indices = np.array([stamps[0], stamps[-1]])
        sequences = preprocess_time(time_indices, 1, self.time_tokenizer)
        sequences = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=self.time_tokenizer.pad_token_id)
        compress_mask = sequences.ne(self.time_tokenizer.pad_token_id)
        questions = preprocess_question([system_prompt + "\n\n" + question], self.time_tokenizer)
        question_ids = torch.nn.utils.rnn.pad_sequence(
            questions, batch_first=True, padding_value=self.time_tokenizer.pad_token_id)
        question_mask = question_ids.ne(self.time_tokenizer.pad_token_id)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).to(self.device)
        stop_ids = self.tokenizer("<|im_end|>").input_ids
        stop = torch.tensor(stop_ids, device=self.device).reshape(-1)
        captured = {}

        def capture_input(module, args, kwargs):
            del module, args
            embeds = kwargs.get("inputs_embeds")
            if embeds is not None and "prompt_tokens" not in captured:
                captured["prompt_tokens"] = int(embeds.shape[1])

        handle = self.model.get_model().register_forward_pre_hook(capture_input, with_kwargs=True)
        try:
            with torch.inference_mode():
                output = self.model.generate(
                    input_ids,
                    images=video.to(self.device, dtype=torch.float16),
                    images_large=video[:, :1].contiguous().to(self.device, dtype=torch.float16),
                    seqs=sequences.to(self.device), compress_mask=compress_mask.to(self.device),
                    qs=question_ids.to(self.device), qs_mask=question_mask.to(self.device),
                    ans_token=self.time_tokenizer(DEFAULT_ANS_TOKEN, return_tensors="pt").input_ids.to(self.device),
                    todo_token=self.time_tokenizer(DEFAULT_TODO_TOKEN, return_tensors="pt").input_ids.to(self.device),
                    q_id=None, insert_position=0, ans_position=[], do_sample=False,
                    max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([StoppingCriteriaSub(stops=[stop])]),
                    use_cache=True, return_dict_in_generate=True,
                )
        finally:
            handle.remove()
        sequences = output.sequences
        raw_text = self.tokenizer.batch_decode(sequences, skip_special_tokens=False)[0].strip()
        text = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)[0].strip()
        token_ids = sequences[0].tolist()
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
        terminated = bool(token_ids and token_ids[-1] in eos_ids)
        if stop_ids and token_ids[-len(stop_ids):] == stop_ids:
            terminated = True
        silent = DEFAULT_SILENT_TOKEN in raw_text
        if silent:
            text = ""
        return {
            "text": text, "finish_reason": "stop" if terminated else "length",
            "prompt_tokens": captured.get("prompt_tokens"),
            "completion_tokens": len(token_ids),
            "preprocessing": {
                "source_revision": "803f6b4634b3232293e89a38bcce9f5815c33ed4",
                "model_revision": "7c5517eaaa4dc3788602af40e077063792c75f6c",
                "attention_backend": "sdpa", "official_stack": False,
                "unique_input_images": 8, "model_frame_slots": 16,
                "slot_to_unique_image": [index for index in range(8) for _ in range(2)],
                "native_sampling": False, "video_clips": 1,
                "path": "official_offline_reaction_with_perception_decision_memory",
                "native_streaming_trigger_exercised": False,
                "reaction_called_once_per_user_query": True,
                "keep_silent_emitted": silent,
                "checkpoint_weight_changes": False,
                "original_image_size": [width, height],
                "upstream_square_resize": [min(width, height), min(width, height)],
                "processor_pixel_values_shape": list(video.shape),
                "clip_time_range_seconds": time_indices.tolist(),
                "input_metadata": frame_metadata,
                "prompt_token_measurement": "first_actual_Reaction_decoder_inputs_embeds_length",
            },
        }
