// Copyright © 2026 Eigen Labs.

import Cmlx
import Foundation
import MLX

/// The process default streams, as opposed to the streams that a task scopes
/// with `Stream.withNewDefaultStream` or `Device.withDefaultDevice`.
///
/// In the pinned mlx-swift, `Stream.gpu` and `Stream.cpu` are the streams of
/// the current task scope. Inside `Stream.withNewDefaultStream(device: .gpu)`,
/// `Stream.gpu` is the new private stream. The fused kernels of this package
/// run only on the process default GPU stream and decline custom streams, so
/// they compare against the streams that this type reads once, from a thread
/// that carries no task-local stream scope.
public enum MLXProcessStreams {

    /// The process default GPU stream.
    public static var gpu: MLX.Stream { processGPU }

    /// The process default CPU stream.
    public static var cpu: MLX.Stream { processCPU }

    /// True when `stream` is the process default GPU stream.
    ///
    /// A CPU stream gives false without reading the GPU stream, so the check is
    /// safe on a machine without a GPU.
    public static func isDefaultGPU(_ stream: StreamOrDevice) -> Bool {
        isGPU(stream) && stream.stream == processGPU
    }

    /// True when the default device is the GPU and operations run on the
    /// process default GPU stream, that is, no task scope set a custom stream.
    public static var isOnDefaultGPUStream: Bool {
        Device.defaultDevice().deviceType == .gpu && isDefaultGPU(.default)
    }

    private static let processGPU: MLX.Stream = unscoped { MLX.Stream.gpu }
    private static let processCPU: MLX.Stream = unscoped { MLX.Stream.cpu }

    private static func isGPU(_ stream: StreamOrDevice) -> Bool {
        var device = mlx_device_new()
        defer { mlx_device_free(device) }
        var type = MLX_CPU
        return mlx_stream_get_device(&device, stream.ctx) == 0
            && mlx_device_get_type(&type, device) == 0 && type == MLX_GPU
    }

    private final class Slot: @unchecked Sendable {
        var stream: MLX.Stream?
    }

    /// Reads `read` on a new thread. Task-local values do not reach a new
    /// thread, so the read sees the process default, not the caller's scope.
    /// `DispatchQueue.sync` is not used because it can run the block inline.
    private static func unscoped(_ read: @escaping @Sendable () -> MLX.Stream) -> MLX.Stream {
        let slot = Slot()
        let done = DispatchSemaphore(value: 0)
        let thread = Thread {
            slot.stream = read()
            done.signal()
        }
        thread.start()
        done.wait()
        guard let stream = slot.stream else {
            preconditionFailure("MLXProcessStreams could not read the process stream")
        }
        return stream
    }
}
