// ExpertStreamingConfig.swift — Swift API for expert streaming configuration
//
// Replaces the EXPERIMENTAL_SSD_STREAM environment variable gate so expert
// streaming can be enabled on iOS (via mmap fallback) without needing
// environment variables, which iOS apps cannot set.
//
// Usage (InferenceEngine.load):
//   ExpertStreamingConfig.shared.activate(modelDirectory: dir, useDirectIO: false)
//
// Usage (SwitchLayers, Load, LayerPartitioning):
//   if ExpertStreamingConfig.shared.isEnabled { ... }

import Foundation

// MARK: — Expert Streaming Mode

public enum ExpertStreamingMode: Sendable {
    /// Expert streaming disabled — all expert weights loaded into RAM at startup.
    case disabled

    /// iOS path: expert weights mmap'd from APFS. The OS page cache serves
    /// pages on first access and evicts them under memory pressure.
    /// Bandwidth ~2–3 GB/s sequential; works entirely within iOS sandbox.
    case mmapPageCache(modelDirectory: URL)

    /// macOS path: expert weights read via pread() directly from NVMe at
    /// ~5 GB/s, bypassing the page cache entirely. Expert tensors never
    /// appear as "resident" — fresh allocator::malloc per expert, freed immediately.
    case directNVMe(modelDirectory: URL)
}

// MARK: — ExpertStreamingConfig

/// Shared configuration for MoE expert streaming.
/// Set `.mode` before calling `loadContainer` / `loadWeights`.
public final class ExpertStreamingConfig: @unchecked Sendable {
    public static let shared = ExpertStreamingConfig()
    private init() {}

    /// Current streaming mode. Thread-safe via MainActor coordination from InferenceEngine.
    public var mode: ExpertStreamingMode = .disabled

    // MARK: — Expert slot-cache budgeting (machine-size aware)

    /// Total byte budget for ALL layers' stacked expert slot buffers combined.
    /// Set by the host (server) from: physicalRAM − OS headroom − resident weights
    /// − KV-cache budget − page-cache reserve. When set, SwitchGLU derives its
    /// per-layer slot count as budget / (moeLayerCountHint × per-slot slab bytes),
    /// so a 48GB machine simply gets fewer slots than a 128GB one instead of
    /// over-allocating. `MLX_MOE_CACHE_SLOTS` still overrides everything.
    public var expertSlotBudgetBytes: Int? = nil

    /// Number of MoE layers that will allocate slot buffers (e.g. 43 for
    /// DeepSeek-V4-Flash). Used with `expertSlotBudgetBytes`.
    public var moeLayerCountHint: Int? = nil

    // MARK: — Convenience accessors

    public var isEnabled: Bool {
        if case .disabled = mode { return false }
        return true
    }

    /// The model directory to use for resolving safetensors files.
    public var modelDirectory: URL? {
        switch mode {
        case .disabled:                          return nil
        case .mmapPageCache(let dir):            return dir
        case .directNVMe(let dir):               return dir
        }
    }

    /// True only when directNVMe mode is active (macOS MLXFast.streamedGatherMM path).
    public var useDirectNVMe: Bool {
        if case .directNVMe = mode { return true }
        return false
    }

    // MARK: — Activation helpers

    /// Activate expert streaming for a model directory.
    /// - Parameters:
    ///   - modelDirectory: The local directory containing model safetensors files.
    ///   - useDirectIO: If true and on macOS, uses direct NVMe pread() (5 GB/s).
    ///                  If false or on iOS, uses mmap page-cache fallback (2–3 GB/s).
    public func activate(modelDirectory: URL, useDirectIO: Bool = false) {
        #if os(macOS)
        if useDirectIO {
            mode = .directNVMe(modelDirectory: modelDirectory)
        } else {
            mode = .mmapPageCache(modelDirectory: modelDirectory)
        }
        #else
        // iOS: always use mmap page-cache — no direct NVMe pread() access
        mode = .mmapPageCache(modelDirectory: modelDirectory)
        #endif
    }

    /// Disable expert streaming and free resources.
    public func deactivate() {
        mode = .disabled
    }
}

// MARK: — Backward Compatibility Shim

extension ExpertStreamingConfig {
    /// Returns the model directory path string if active, for use in C-level APIs.
    /// This replaces `ProcessInfo.processInfo.environment["EXPERIMENTAL_SSD_STREAM"]`.
    public var legacyEnvPath: String? {
        modelDirectory?.path
    }
}
