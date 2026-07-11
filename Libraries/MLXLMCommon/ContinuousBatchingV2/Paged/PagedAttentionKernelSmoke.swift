// PagedAttentionKernelSmoke.swift
//
// Pre-traffic compilation/dispatch coverage for paged Metal kernels.

import Foundation
import MLX
import MLXRandom

public struct PagedAttentionKernelSmokeShape: Hashable, Sendable {
    public let headDim: Int
    public let kvHeads: Int
    public let queryHeads: Int
    public let hasSinks: Bool
    public let hasWrite: Bool

    public init(
        headDim: Int,
        kvHeads: Int,
        queryHeads: Int,
        hasSinks: Bool,
        hasWrite: Bool
    ) {
        self.headDim = headDim
        self.kvHeads = kvHeads
        self.queryHeads = queryHeads
        self.hasSinks = hasSinks
        self.hasWrite = hasWrite
    }

    public var argumentValue: String {
        [
            headDim,
            kvHeads,
            queryHeads,
            hasSinks ? 1 : 0,
            hasWrite ? 1 : 0,
        ].map(String.init).joined(separator: ":")
    }

    public init(argumentValue: String) throws {
        let fields = argumentValue.split(
            separator: ":",
            omittingEmptySubsequences: false)
        guard fields.count == 5,
            let headDim = Int(fields[0]),
            let kvHeads = Int(fields[1]),
            let queryHeads = Int(fields[2]),
            let sinks = Int(fields[3]),
            let write = Int(fields[4]),
            (sinks == 0 || sinks == 1),
            (write == 0 || write == 1)
        else {
            throw PagedAttentionKernelSmokeError.invalidShape(argumentValue)
        }
        self.init(
            headDim: headDim,
            kvHeads: kvHeads,
            queryHeads: queryHeads,
            hasSinks: sinks == 1,
            hasWrite: write == 1)
    }
}

public enum PagedAttentionKernelSmokeError: Error, CustomStringConvertible {
    case invalidShape(String)
    case ineligibleShape(String)

    public var description: String {
        switch self {
        case .invalidShape(let value):
            return "invalid paged-kernel smoke shape \(value)"
        case .ineligibleShape(let reason):
            return "ineligible paged-kernel smoke shape: \(reason)"
        }
    }
}

extension PagedAttentionKernel {
    /// Canonical production GPT-OSS shape. Full and sliding layers share
    /// the same compiled kernel specialization.
    public static let gptOSSRuntimeSmokeShapes = [
        PagedAttentionKernelSmokeShape(
            headDim: 64,
            kvHeads: 8,
            queryHeads: 64,
            hasSinks: true,
            hasWrite: true)
    ]

    public static func smokeShapes(
        layerKinds: [CBv2LayerKind]
    ) -> [PagedAttentionKernelSmokeShape] {
        Array(Set(layerKinds.map {
            PagedAttentionKernelSmokeShape(
                headDim: $0.headDim,
                kvHeads: $0.kvHeads,
                queryHeads: $0.queryHeads,
                hasSinks: $0.hasSinks,
                hasWrite: $0.sharesKVWithLayer == nil)
        })).sorted {
            (
                $0.headDim,
                $0.kvHeads,
                $0.queryHeads,
                $0.hasSinks ? 1 : 0,
                $0.hasWrite ? 1 : 0
            ) < (
                $1.headDim,
                $1.kvHeads,
                $1.queryHeads,
                $1.hasSinks ? 1 : 0,
                $1.hasWrite ? 1 : 0
            )
        }
    }

    /// Catchable resource preflight used by backend construction and
    /// packaged-artifact verification. A packaged process searches only
    /// its sealed app resources.
    public static func validateRuntimeResources() throws {
        _ = try PagedAttentionResources.loadSourceForCurrentProcess()
    }

    /// Pre-JIT every paged kernel specialization a model can dispatch:
    /// prefill/adoption bulk-write, decode part (fused-write or borrowing),
    /// and decode merge (with the model's sink specialization). Every output
    /// is evaluated so compilation and dispatch complete before traffic.
    public static func runtimeSmoke(
        shapes: [PagedAttentionKernelSmokeShape] = gptOSSRuntimeSmokeShapes
    ) throws {
        let source = try PagedAttentionResources.loadSourceForCurrentProcess()
        try runtimeSmoke(source: source, shapes: shapes)
    }

    static func runtimeSmokeForTesting(
        searchRoots: [URL],
        shapes: [PagedAttentionKernelSmokeShape] = gptOSSRuntimeSmokeShapes
    ) throws {
        let source = try PagedAttentionResources.loadSource(roots: searchRoots)
        try runtimeSmoke(source: source, shapes: shapes)
    }

    private static func runtimeSmoke(
        source: String,
        shapes: [PagedAttentionKernelSmokeShape]
    ) throws {
        guard !shapes.isEmpty else {
            throw PagedAttentionKernelSmokeError.invalidShape("empty")
        }

        let pageSize = CBv2PagedDefaults.pageSize
        let slots = MLXArray([Int32](repeating: 0, count: 8))
        for shape in shapes {
            guard shape.kvHeads > 0,
                shape.queryHeads > 0,
                shape.queryHeads % shape.kvHeads == 0
            else {
                throw PagedAttentionKernelSmokeError.invalidShape(
                    shape.argumentValue)
            }
            let gqa = shape.queryHeads / shape.kvHeads
            if let reason = ineligibilityReason(
                headDim: shape.headDim,
                gqa: gqa)
            {
                throw PagedAttentionKernelSmokeError.ineligibleShape(reason)
            }

            let kSlab = MLXArray.zeros(
                [1, shape.kvHeads, pageSize, shape.headDim],
                dtype: .float16)
            let vSlab = MLXArray.zeros(
                [1, shape.kvHeads, pageSize, shape.headDim],
                dtype: .float16)
            let writeTile = MLXArray.zeros(
                [shape.kvHeads, 1, shape.headDim],
                dtype: .float16)
            let fence = bulkWrite(
                kSlab: kSlab,
                vSlab: vSlab,
                keys: writeTile,
                values: writeTile,
                slots: slots,
                prevFence: MLXArray.zeros([1], dtype: .int32),
                pageSize: pageSize,
                kernelSource: source)
            eval(fence)

            let queries = MLXArray.zeros(
                [1, shape.queryHeads, 1, shape.headDim],
                dtype: .float16)
            let decodeTile = MLXArray.zeros(
                [1, shape.kvHeads, shape.headDim],
                dtype: .float16)
            let tables = MLXArray([Int32](repeating: 0, count: 8))
                .reshaped([1, 8])
            let seqinfo = MLXArray(
                [Int32(0), 1, 1, 0, 0, 0, 0, 0],
                [1, 8])
            let sinks = shape.hasSinks
                ? MLXRandom.normal(
                    [max(shape.queryHeads, 8)],
                    dtype: .float32)
                : nil
            if let sinks {
                eval(sinks)
            }
            let result = decode(
                queries: queries,
                newKeys: shape.hasWrite ? decodeTile : nil,
                newValues: shape.hasWrite ? decodeTile : nil,
                kSlab: kSlab,
                vSlab: vSlab,
                tables: tables,
                seqinfo: seqinfo,
                maxAttendLength: 1,
                sinks: sinks,
                params: MLXArray(
                    [Float(1), 0.125, 0, 0, 0, 0, 0, 0]),
                softcap: false,
                pageSize: pageSize,
                writeFence: fence,
                kernelSource: source)
            if let nextFence = result.nextWriteFence {
                eval(result.out, nextFence)
            } else {
                eval(result.out)
            }
        }
    }
}
