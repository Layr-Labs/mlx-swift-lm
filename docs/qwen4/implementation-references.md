# Qwen4 implementation references and notices

The native Swift port credits the following implementation references. This
notice preserves upstream notices for adapted portions; it does not relicense
unrelated files or declare experimental paths qualified for production.

## oMLX Fusion

The Qwen4 model, PLE, gathered attention, MTP and profiling modules identify
oMLX Fusion references in their source comments, including
`Libraries/MLXLLM/Models/Qwen4Exp.swift`, `Qwen4ExpPLE.swift`,
`Qwen4ExpQSA.swift`, `Qwen4ExpQSASteel.swift`, `Qwen4ExpDecodeProfile.swift`,
and `Qwen4ExpLayerSubmission.swift`. The last adapts early layer submission
to native paged fault checks and engine-owned deferred-fill/retirement scopes.

Reference: <https://github.com/jonathan308/omlx-fusion>.

Copyright 2025 oMLX contributors.

Adapted oMLX portions are provided under the Apache License, Version 2.0.
The complete license is in [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0). Swift translation, native
paged storage integration, explicit state/ownership validation and local
qualification are modifications, not an assertion that the upstream code was
distributed in this form. Model preprocessing references have their own notice
in [preprocessing attribution](preprocessing.md).

## mlx-serve

The verify-width gathered-attention reference in `Qwen4ExpQSA.swift` and the
deferred PLE scheduling reference in `Qwen4ExpPLE.swift` credit David Dalcu's
mlx-serve implementation. These references include experimental or compatibility
paths; attribution does not imply every referenced path is enabled by default.

Reference: <https://github.com/ddalcu/mlx-serve>.

Copyright (c) 2026 David Dalcu.

## MLX Steel helpers

`Qwen4ExpQSASteel.swift` and `Qwen4ExpAffineQMM.swift` identify MLX Steel
matrix/tile helpers adapted for the native custom-kernel interface.

Reference: <https://github.com/ml-explore/mlx>.

Copyright © 2023 Apple Inc.

The packaged `Libraries/MLXLMCommon/Resources/Qwen4Metal/` fragments are exact
`R"preamble(...)preamble"` payloads from `gemm.cpp`, `quantized_utils.cpp`,
and `quantized.cpp` in mlx-swift revision
`6d6796d7a81b656d2749d39067e0a6bea2bc2986`, whose core pin is
`3fa8f25e6451174d7b06be372c3a24272b77d88e`. Their original Apple copyright
notices and the core MIT license are preserved. Packaging changes resource
lookup only; it does not change the generated Metal source. The raw resources
are copied by SwiftPM and JIT-compiled by the existing native Qwen kernels.

## MIT permission notice for the MIT-licensed portions above

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
