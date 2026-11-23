// Copyright © 2026 Eigen Labs.

import MLXLMCommon

/// Image requests on the VLM stay target-only (ordinary recurrent token and
/// embedding forwards, never drafts, verify, carry or assistant state) while
/// text requests beside them keep drafting with the DSpark drafter. Exact for
/// the same reason as the text model's opt-in (`Inkling2Model`): row-exact
/// decode and one prefill step for every prompt chunk.
extension Inkling2VLM: CBv2RecurrentMTPRequestScopedMediaFallback {}
