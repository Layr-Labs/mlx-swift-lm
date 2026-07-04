// Per-thread MLX default-stream pinning.

import Cmlx
import Foundation
import MLX

/// Pin the calling thread's MLX default streams to the process-global
/// thread-unsafe streams (`Stream.gpu` / `Stream.cpu` in mlx-swift's
/// Stream.swift).
///
/// WHY: MLX 0.32 made default streams thread-local (mlx/stream.cpp:
/// `default_stream_storage` is `thread_local`, lazily calling `new_stream`).
/// The first eval on ANY unpinned thread therefore mints a fresh
/// Stream(gpu, N) whose Metal command encoder is registered only in that
/// thread's `thread_local` encoder map — `eval_impl` (mlx/transforms.cpp)
/// consults `default_stream(default_device())` for its synchronizer/event
/// bookkeeping even when every user op carries an explicit stream. Once
/// state tagged with that per-thread stream is observed by a later eval on
/// a DIFFERENT thread, MLX aborts with "There is no Stream(gpu, N) in
/// current thread" (neither that thread's encoder map nor the global one
/// knows the stream).
///
/// mlx-swift's process-global `Stream.gpu`/`Stream.cpu` exist precisely to
/// restore single-default-stream semantics (see the rationale comment in
/// Stream.swift), but they only help threads whose *default* streams point
/// at them. Generation hops threads by construction — `TokenIterator` is
/// prepared on the `ModelContainer` actor thread and iterated on the
/// `generateLoopTask` task thread — so every thread that evals must pin
/// first. Idempotent, two thread-local writes; safe to call per step.
package func pinThreadDefaultStreamsToGlobal() {
    mlx_set_default_stream(StreamOrDevice.gpu.ctx)
    mlx_set_default_stream(StreamOrDevice.cpu.ctx)
}
