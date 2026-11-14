"""Golden layout of Transformers' own Qwen3.5 processor on fixed synthetic clips.

Runs Hugging Face Transformers 5.8.0 `Qwen3VLProcessor` (the processor class the
Qwen3.5/3.6/3.8 checkpoints name) with the checkpoint's own tokenizer, chat
template and image/video preprocessor configs, on clips described by frame
count, fps and size. It records what the Swift port must reproduce: the
expanded token ids, `video_grid_thw`, the sampled frame indices, and Qwen3.5's
own `get_rope_index` positions and delta (called without weights).

The compact fixture carries a tokenizer table instead of the 11 MB tokenizer:
the template's ids, the special tokens, and the ids of every text segment
between special tokens (Transformers tokenizes those segments independently).

Usage (Python 3.10+, transformers==5.8.0, torch, numpy; offline):
  python qwen35_video_layout_golden.py <checkpoint dir> <full.json> \
      Tests/MLXLMTests/Resources/qwen35-video-layout-golden.json
The checkpoint dir needs only the tokenizer, chat template, config.json and
preprocessor_config.json / video_preprocessor_config.json.
"""

import hashlib
import json
import re
import sys
from types import SimpleNamespace

import numpy as np
import torch
import transformers
from transformers import AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
from transformers.video_utils import VideoMetadata

# name, frames, fps, height, width, max_frames, clip pixel budget (size.longest_edge)
CASES = [
    ("k32-720p-budget256", 32, 1.0, 720, 1280, 32, 256 * 2048),
    ("k64-cap32-budget512", 64, 1.0, 720, 1280, 32, 512 * 2048),
    ("odd7-unbound", 7, 1.0, 180, 320, 64, 8192 * 2048),
    ("odd9-ceil-budget256", 9, 1.0, 720, 1280, 64, 256 * 2048),
    ("30fps-portrait-budget1024", 300, 30.0, 320, 180, 64, 1024 * 2048),
    ("short-min4", 5, 10.0, 64, 64, 64, 1024 * 2048),
    ("reference-defaults-small", 64, 1.0, 144, 256, 768, 25165824),
]

SPECIAL = re.compile(r"(<\|[a-z_]+\|>|</?think>)")


def run_length(ids):
    runs = []
    for token in ids:
        if runs and runs[-1][0] == token:
            runs[-1][1] += 1
        else:
            runs.append([token, 1])
    return runs


def main(checkpoint: str, full_path: str, fixture_path: str) -> None:
    processor = AutoProcessor.from_pretrained(checkpoint)
    tokenizer = processor.tokenizer
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": "What happens in the video?"},
            ],
        }
    ]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model = SimpleNamespace(
        config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    )
    model.get_vision_position_ids = (
        lambda *args, **kwargs: Qwen3_5Model.get_vision_position_ids(model, *args, **kwargs)
    )

    full, compact = [], []
    special, segments = {}, {}
    for name, frames, fps, height, width, max_frames, longest_edge in CASES:
        video = np.zeros((frames, height, width, 3), dtype=np.uint8)
        metadata = VideoMetadata(
            total_num_frames=frames,
            fps=fps,
            width=width,
            height=height,
            duration=frames / fps,
            frames_indices=list(range(frames)),
        )
        # Transformers 5.8 samples with the processor's own `max_frames`
        # (a call kwarg does not reach `sample_frames`).
        processor.video_processor.max_frames = max_frames
        out = processor(
            text=[rendered],
            videos=[video],
            video_metadata=[metadata],
            do_sample_frames=True,
            fps=2.0,
            size={"shortest_edge": 4096, "longest_edge": longest_edge},
            return_metadata=True,
            return_mm_token_type_ids=True,
            return_tensors="pt",
        )
        ids = out["input_ids"]
        grid = out["video_grid_thw"]
        positions, delta = Qwen3_5Model.get_rope_index(
            model,
            input_ids=ids,
            mm_token_type_ids=out["mm_token_type_ids"],
            image_grid_thw=None,
            video_grid_thw=grid.clone(),
            attention_mask=None,
        )
        flat = positions[:, 0, :].to(torch.int32).numpy().astype("<i4")
        text = tokenizer.decode(ids[0].tolist(), skip_special_tokens=False)
        for piece in SPECIAL.split(text):
            if not piece:
                continue
            if SPECIAL.fullmatch(piece):
                special[piece] = tokenizer.convert_tokens_to_ids(piece)
            else:
                segments[piece] = tokenizer.encode(piece, add_special_tokens=False)
        sampled = [int(i) for i in out["video_metadata"][0].frames_indices]
        case = {
            "name": name,
            "frames": frames,
            "fps": fps,
            "height": height,
            "width": width,
            "max_frames": max_frames,
            "max_pixels": longest_edge,
            "sampled_indices": sampled,
            "grid": [int(v) for v in grid[0].tolist()],
            "token_count": int(ids.shape[1]),
            "ids_rle": run_length(ids[0].tolist()),
            "timestamps": re.findall(r"<[0-9.]+ seconds>", text),
            "positions_sha256": hashlib.sha256(flat.tobytes()).hexdigest(),
            "delta": int(delta.reshape(-1)[0]),
        }
        compact.append(case)
        full.append(dict(case, positions=flat.tolist(), expanded_text=text))
        print(name, "frames", len(sampled), "grid", case["grid"], "tokens", case["token_count"],
              "delta", case["delta"])

    header = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "processor": "Qwen3VLProcessor + Qwen3VLVideoProcessor, EigenLabs/Qwen3.8-27B-4bit-mtp configs",
        "rendered_template": rendered,
        "rendered_template_ids": tokenizer.encode(rendered, add_special_tokens=False),
        "special_tokens": special,
        "segments": segments,
    }
    # The template's own segments, so the fixture tokenizer can encode it too.
    for piece in SPECIAL.split(rendered):
        if piece and not SPECIAL.fullmatch(piece):
            segments[piece] = tokenizer.encode(piece, add_special_tokens=False)
        elif piece:
            special[piece] = tokenizer.convert_tokens_to_ids(piece)
    with open(full_path, "w") as f:
        json.dump(dict(header, cases=full), f, indent=1, ensure_ascii=False)
    with open(fixture_path, "w") as f:
        json.dump(dict(header, cases=compact), f, indent=1, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
