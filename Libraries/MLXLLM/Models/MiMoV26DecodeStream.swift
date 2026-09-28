// Copyright © 2026 Eigen Labs.

import Cmlx
import MLX

/// Resolve the actual operation stream, including a task-local override.
/// Device.defaultDevice alone does not describe StreamOrDevice.default.
enum MiMoV26DecodeStream {
    static func deviceType(of stream: StreamOrDevice) -> DeviceType? {
        var device = mlx_device_new()
        defer { mlx_device_free(device) }
        var type = MLX_CPU
        guard mlx_stream_get_device(&device, stream.ctx) == 0,
              mlx_device_get_type(&type, device) == 0 else { return nil }
        if type == MLX_GPU { return .gpu }
        if type == MLX_CPU { return .cpu }
        return nil
    }
}
