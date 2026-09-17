# Qwen4 preprocessing reference attribution

`Libraries/MLXVLM/Models/Qwen4ExpMediaGeometry.swift`,
`Qwen4ExpVideoPrompt.swift`, `Qwen4ExpVideoConfiguration.swift`, and
`Qwen4ExpVideoPreparation.swift` adapt the resize, processor selection,
frame-index sampling and temporal timestamp algorithms from Hugging Face
Transformers v5.8.0, commit `049d2bf1220747b6d39e2a978b9f5fe0defa1dca`:

- `src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py`
- `src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py`
- `src/transformers/models/qwen3_vl/processing_qwen3_vl.py`
- `src/transformers/video_processing_utils.py`

The Qwen4 wrapper's per-temporal-group position-grid expansion follows
`src/transformers/models/qwen3_vl/modeling_qwen3_vl.py`. AVFoundation extraction
and memory-backed ownership in `Qwen4ExpVideoSampler.swift` are native Swift
implementation work, not an invocation of Python or an external decoder process.

Copyright 2024 The Qwen team, Alibaba Group and The HuggingFace Inc. team.
All rights reserved.
Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.

These adaptations are provided under Apache License 2.0; the complete license
is in [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0). Modifications include translation to Swift, explicit
overflow/metadata validation, and separation of geometry from allocation and
memory admission. This notice does not relicense unrelated files or establish
that all preprocessing integration and model qualification gates have passed.
