// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM

/// Deterministic SplitMix64 inputs shared by the parity tests and benchmark.
private struct SelectionInputs {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func int(_ bound: Int) -> Int { Int(next() % UInt64(bound)) }
    mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 24) }

    static let minimum = -Float.greatestFiniteMagnitude
    static let specials: [Float] = [
        0, -0.0, .infinity, -.infinity, .nan, -.nan, minimum, 1, -1,
        Float.leastNonzeroMagnitude, -Float.leastNonzeroMagnitude, Float.leastNormalMagnitude,
        Float.greatestFiniteMagnitude, Float(bitPattern: 0x7FC0_0001),
    ]

    /// 0 continuous, 1 small integers, 2 specials, 3 finite-minimum future
    /// pools, 4 float16-rounded, 5 bfloat16-rounded.
    mutating func value(style: Int) -> Float {
        switch style {
        case 0: return unit() * 2 - 1
        case 1: return Float(int(8)) - 4
        case 2: return int(4) == 0 ? Self.specials[int(Self.specials.count)] : Float(int(3))
        case 3: return int(3) == 0 ? Self.minimum : unit()
        case 4: return Float(Float16(unit() * 4))
        default:
            let bits = (unit() * 8 - 4).bitPattern
            return Float(bitPattern: (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) & 0xFFFF_0000)
        }
    }

    mutating func mask(style: Int, batch: Int, pools: Int) -> [Bool] {
        (0 ..< batch * pools).map { index in
            switch style {
            case 0: return true
            case 1: return int(10) != 0
            case 2: return index % pools < pools - 1 || int(2) == 0
            default: return index / pools == 0 ? index % pools < pools / 2 : int(3) == 0
            }
        }
    }
}

private struct SelectionCase: CustomStringConvertible {
    let label: String
    let batch: Int
    let pools: Int
    let values: [Float]
    let complete: [Bool]
    let count: Int

    var description: String { "\(label) B=\(batch) P=\(pools) K=\(count)" }
    var hasSubnormal: Bool { values.contains { $0.isSubnormal } }

    func keep(_ pool: Int) -> Bool { (0 ..< batch).contains { complete[$0 * pools + pool] } }

    /// Score-only reference order as an unsigned key: NaNs tie at the top and
    /// -0 joins +0. Zero marks a pool no row keeps.
    func key(row: Int, pool: Int) -> UInt32 {
        guard keep(pool) else { return 0 }
        let x = values[row * pools + pool]
        if x.isNaN { return 0xFFFF_FFFF }
        let bits: UInt32 = x == 0 ? 0 : x.bitPattern
        return bits & 0x8000_0000 != 0 ? ~bits : bits | 0x8000_0000
    }

    /// The K-th key, or nil when every retained pool (or none) is selected.
    func threshold(row: Int) -> (key: UInt32, tied: Bool)? {
        let keys = (0 ..< pools).map { key(row: row, pool: $0) }.filter { $0 != 0 }
            .sorted(by: >)
        let selected = min(count, keys.count)
        guard selected > 0, selected < keys.count else { return nil }
        return (keys[selected - 1], keys[selected - 1] == keys[selected])
    }

    func thresholdMembers(row: Int) -> [Int32] {
        let bound = threshold(row: row)?.key ?? 1
        let members = (0 ..< pools).filter { key(row: row, pool: $0) >= bound }.map(Int32.init)
        return members + Array(repeating: -1, count: min(count, pools) - members.count)
    }
}

extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextParallelSparseSelectionTests {
        private typealias Strategy = GLM5NextSparseAttention.SelectionStrategy

        private func fence(_ arrays: [MLXArray]) throws {
            try MLX.withError { error in
                eval(arrays)
                StreamOrDevice.default.stream.synchronize()
                try error.check()
            }
        }

        /// Raw per-row output in kernel order; pool-count padding becomes -1.
        private func raw(
            _ input: SelectionCase, strategy: Strategy, dtype: DType = .float32,
            requestedCount: Int? = nil
        ) throws -> [[Int32]] {
            let scores = MLXArray(input.values, [input.batch, input.pools]).asType(dtype)
            let complete = MLXArray(input.complete, [input.batch, input.pools])
            let result = try GLM5NextSparseSelection.select(
                scores: scores, completePools: complete,
                requestedCount: requestedCount ?? input.count, strategy: strategy)
            try fence([result])
            let width = min(input.count, input.pools)
            #expect(result.dtype == .int32 && result.shape == [input.batch, width])
            let ids = result.asArray(Int32.self).map { $0 == Int32(input.pools) ? -1 : $0 }
            return (0 ..< input.batch).map { Array(ids[($0 * width) ..< (($0 + 1) * width)]) }
        }

        /// CPU reference helper: the pinned diagnostic emulation on a CPU stream.
        private func cpuReference(_ input: SelectionCase) throws -> [[Int32]] {
            try Stream.withNewDefaultStream(device: .cpu) {
                try raw(input, strategy: .reference).map { $0.sorted() }
            }
        }

        private static func randomCases(seed: UInt64, count: Int, maximumPools: Int)
            -> [SelectionCase]
        {
            var inputs = SelectionInputs(state: seed)
            let poolChoices = [
                1, 2, 3, 5, 7, 8, 9, 31, 64, 127, 128, 129, 255, 256, 257, 1000,
                2047, 4096, 8191, 32767, 32768,
            ].filter { $0 <= maximumPools }
            let countChoices = [1, 2, 3, 4, 7, 8, 63, 64, 65, 255, 256, 511, 512]
            return (0 ..< count).map { index in
                let pools = poolChoices[inputs.int(poolChoices.count)]
                let batch = 1 + inputs.int(3)
                let style = inputs.int(6)
                var values = (0 ..< batch * pools).map { _ in inputs.value(style: style) }
                if inputs.int(3) == 0 {
                    let tie = Float(inputs.int(5))
                    for _ in 0 ... inputs.int(min(pools, 16)) {
                        values[inputs.int(values.count)] = tie
                    }
                }
                return SelectionCase(
                    label: "seed\(seed)#\(index) style\(style)", batch: batch, pools: pools,
                    values: values,
                    complete: inputs.mask(style: inputs.int(4), batch: batch, pools: pools),
                    count: countChoices[inputs.int(countChoices.count)])
            }
        }

        /// The fast path's correctness premise, checked against the existing
        /// CPU emulation: an untied K-th score makes membership unique, and a
        /// tied one still yields a valid top-K. Runs on the CPU stream only.
        @Test func referenceMembershipIsTheThresholdSetWhenUntied() throws {
            for input in Self.randomCases(seed: 11, count: 160, maximumPools: 8191)
                + Self.randomCases(seed: 12, count: 4, maximumPools: 32768)
            {
                let expected = try cpuReference(input)
                for row in 0 ..< input.batch {
                    let members = expected[row]
                    guard let threshold = input.threshold(row: row), threshold.tied else {
                        // `cpuReference` sorts each row, so its -1 padding
                        // comes first; compare the threshold set the same way.
                        #expect(
                            members == input.thresholdMembers(row: row).sorted(),
                            "\(input) row \(row)")
                        continue
                    }
                    let keys = members.filter { $0 >= 0 }.map {
                        input.key(row: row, pool: Int($0))
                    }
                    let above = (0 ..< input.pools).filter {
                        input.key(row: row, pool: $0) > threshold.key
                    }
                    #expect(keys.allSatisfy { $0 >= threshold.key }, "\(input) row \(row)")
                    #expect(Set(above.map(Int32.init)).isSubset(of: Set(members)))
                    #expect(members.count == min(input.count, input.pools))
                    #expect(members.allSatisfy { $0 >= 0 }, "\(input) row \(row)")
                }
            }
        }

        private func expectParity(_ input: SelectionCase, dtype: DType = .float32) throws {
            let reference = try raw(input, strategy: .reference, dtype: dtype)
            let parallel = try raw(input, strategy: .parallelThreshold, dtype: dtype)
            let cpu = dtype == .float32 && !input.hasSubnormal ? try cpuReference(input) : nil
            for row in 0 ..< input.batch {
                #expect(parallel[row].sorted() == reference[row].sorted(), "\(input) row \(row)")
                if let cpu { #expect(parallel[row].sorted() == cpu[row], "\(input) row \(row)") }
                // Subnormal keys depend on the GPU comparison mode; only the
                // device reference is authoritative for those rows.
                guard dtype == .float32, !input.hasSubnormal else { continue }
                if input.threshold(row: row)?.tied == true {
                    // The fallback is the reference row body, order included.
                    #expect(parallel[row] == reference[row], "\(input) tied row \(row)")
                } else {
                    // The parallel emission is ascending with padding last.
                    #expect(
                        parallel[row] == input.thresholdMembers(row: row), "\(input) row \(row)")
                }
            }
        }

        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            @Test(arguments: 0 ..< 8)
            func randomizedParallelMatchesBothReferences(seedIndex: Int) throws {
                try Device.withDefaultDevice(.gpu) { () throws -> Void in
                    for input in Self.randomCases(
                        seed: 100 + UInt64(seedIndex), count: 48,
                        maximumPools: seedIndex < 6 ? 8191 : 32768)
                    {
                        try expectParity(input)
                    }
                }
            }

            @Test func adversarialTieBoundaries() throws {
                try Device.withDefaultDevice(.gpu) { () throws -> Void in
                    let minimum = -Float.greatestFiniteMagnitude
                    func single(_ label: String, _ values: [Float], count: Int, mask: [Bool]? = nil)
                        -> SelectionCase
                    {
                        .init(
                            label: label, batch: 1, pools: values.count, values: values,
                            complete: mask ?? Array(repeating: true, count: values.count),
                            count: count)
                    }
                    let ramp = (0 ..< 4096).map { Float($0 % 1024) }
                    var cases: [SelectionCase] = [
                        single(
                            "all equal, nth branch", Array(repeating: 0, count: 2048), count: 512),
                        single(
                            "all equal, heap branch", Array(repeating: 1, count: 32768), count: 512),
                        // Exactly K pools tie at the K-th value: unique membership.
                        single(
                            "tie fully inside",
                            Array(repeating: 0, count: 1000)
                                + Array(repeating: 5, count: 512), count: 512),
                        // One tied pool beyond K straddles the boundary.
                        single(
                            "tie straddles by one",
                            Array(repeating: 0, count: 1000)
                                + Array(repeating: 5, count: 513), count: 512),
                        // Ties directly below an untied K-th value.
                        single(
                            "tie below boundary",
                            Array(repeating: 0, count: 1000)
                                + Array(repeating: 5, count: 511) + [4], count: 512),
                        single("ramp with repeats", ramp, count: 256),
                        single("ramp with repeats K=1", ramp, count: 1),
                        // Prefill: fewer eligible pools than K, futures at the minimum.
                        single(
                            "finite-minimum futures",
                            (0 ..< 3000).map {
                                $0 < 300 ? Float($0) : minimum
                            }, count: 512),
                        single(
                            "NaN group at boundary",
                            Array(repeating: 0, count: 600)
                                + Array(repeating: .nan, count: 520), count: 512),
                        single(
                            "NaN group inside",
                            Array(repeating: 0, count: 600)
                                + Array(repeating: .nan, count: 500) + [-.nan], count: 512),
                        single(
                            "signed zeros",
                            (0 ..< 1500).map { $0 % 2 == 0 ? 0 : -0.0 }
                                + Array(repeating: 1, count: 200), count: 512),
                        single(
                            "infinities",
                            Array(repeating: .infinity, count: 300)
                                + Array(repeating: -.infinity, count: 300)
                                + Array(repeating: 0, count: 300),
                            count: 512),
                        single("one pool", [3], count: 512),
                        single("short constant-space axis", [2, 1, 2, 1, 2], count: 2),
                        single(
                            "nothing complete", [1, 2, 3], count: 2, mask: [false, false, false]),
                        single(
                            "one incomplete pool", (0 ..< 700).map { Float($0 % 7) }, count: 512,
                            mask: Array(repeating: true, count: 699) + [false]),
                    ]
                    for pools in [32767, 32768] {
                        // Both sides of the semantic 64*K branch at full K.
                        cases.append(
                            single("ascending \(pools)", (0 ..< pools).map(Float.init), count: 512))
                        cases.append(
                            single(
                                "tied tail \(pools)",
                                (0 ..< pools).map { $0 < pools - 600 ? Float(0) : 1 }, count: 512))
                    }
                    cases.append(
                        single(
                            "subnormals",
                            (0 ..< 900).map { index -> Float in
                                let magnitude = Float(bitPattern: UInt32(index % 300 + 1))
                                return index % 2 == 0 ? magnitude : -magnitude
                            }, count: 512))
                    // One batch: a tied row, an untied row and a finite-minimum
                    // row. Each row keeps a different pool subset; the selector
                    // retains their union, and some pools no row keeps.
                    let pools = 1024
                    cases.append(
                        .init(
                            label: "mixed batch", batch: 3, pools: pools,
                            values: Array(repeating: 0, count: pools)
                                + (0 ..< pools).map(Float.init)
                                + (0 ..< pools).map { $0 < 900 ? minimum : Float($0) },
                            complete: (0 ..< 3 * pools).map { index in
                                let pool = index % pools
                                switch index / pools {
                                case 0: return pool % 5 != 4
                                case 1: return pool % 7 != 6
                                default: return pool < 512
                                }
                            },
                            count: 100))
                    for input in cases { try expectParity(input) }
                }
            }

            @Test func floatingDtypesAndBudgetCount() throws {
                try Device.withDefaultDevice(.gpu) { () throws -> Void in
                    var inputs = SelectionInputs(state: 7)
                    for dtype: DType in [.float16, .bfloat16] {
                        for pools in [9, 700, 5000] {
                            let input = SelectionCase(
                                label: "\(dtype)", batch: 2, pools: pools,
                                values: (0 ..< 2 * pools).map { _ in inputs.unit() * 3 },
                                complete: Array(repeating: true, count: 2 * pools), count: 64)
                            try expectParity(input, dtype: dtype)
                        }
                    }
                    // selectTokens passes the token budget, not min(budget, pools).
                    let short = SelectionCase(
                        label: "budget", batch: 1, pools: 40,
                        values: (0 ..< 40).map { Float($0 % 9) },
                        complete: Array(repeating: true, count: 40), count: 40)
                    let budget = try raw(short, strategy: .parallelThreshold, requestedCount: 512)
                    let exact = try raw(short, strategy: .reference)
                    #expect(budget.map { $0.sorted() } == exact.map { $0.sorted() })
                }
            }

            /// The pinned cases of GLM5NextSparseSelectionTests, under the candidate.
            @Test func pinnedReferenceCasesUnderParallelStrategy() throws {
                try Device.withDefaultDevice(.gpu) { () throws -> Void in
                    let minimum = -Float.greatestFiniteMagnitude
                    let cases: [([Float], [Int32])] = [
                        ([0, 0, 0, 1, 0, minimum], [1, 3]),
                        ([-2, 1, -3, 0, 0, -1, minimum, minimum], [1, 4]),
                        ([0, -3, -2, -1, -2, 0, 0, minimum], [0, 6]),
                        ([0, -3, -2, -1, -2, 0, 0, 0], [0, 6]),
                        ([0, -1, 0, 1, -1, -1], [2, 3]),
                        ([0, 1, 0, -1, -1, -1], [0, 1]),
                        ([0, -.infinity, .infinity, .nan], [2, 3]),
                        ([0, -0.0, 0, -0.0], [0, 1]),
                    ]
                    for (values, expected) in cases {
                        let input = SelectionCase(
                            label: "pinned", batch: 1, pools: values.count, values: values,
                            complete: Array(repeating: true, count: values.count), count: 2)
                        #expect(
                            try raw(input, strategy: .parallelThreshold)[0].sorted() == expected)
                    }
                    let lateWinner = Array(repeating: Float(0), count: 127) + [1]
                    var retained = Array(repeating: Float(0), count: 128)
                    retained[126] = 1
                    for (values, mask, expected): ([Float], [Bool], [Int32]) in [
                        (lateWinner, Array(repeating: true, count: 128), [0, 127]),
                        (retained, Array(repeating: true, count: 128), [0, 126]),
                        (retained, Array(repeating: true, count: 127) + [false], [1, 126]),
                    ] {
                        let input = SelectionCase(
                            label: "branch", batch: 1, pools: values.count, values: values,
                            complete: mask, count: 2)
                        #expect(
                            try raw(input, strategy: .parallelThreshold)[0].sorted() == expected)
                    }
                }
            }

            /// The parallel pipeline is specialized only on `Count` and always
            /// dispatches 256-thread groups. MLX refuses such a dispatch with an
            /// error, never a fallback, when register pressure (including the
            /// inlined reference row) lowers the pipeline's thread limit. This
            /// pins the production pipeline (budget 2048, so K = 512) at the
            /// largest history, with untied, tied and random rows.
            @Test func productionPipelineAdmitsItsThreadgroup() throws {
                try Device.withDefaultDevice(.gpu) { () throws -> Void in
                    var inputs = SelectionInputs(state: 29)
                    let pools = 32768
                    let input = SelectionCase(
                        label: "production pipeline", batch: 3, pools: pools,
                        values: (0 ..< pools).map(Float.init)
                            + Array(repeating: 1, count: pools)
                            + (0 ..< pools).map { _ in inputs.unit() },
                        complete: Array(repeating: true, count: 3 * pools), count: 512)
                    // `raw` evaluates under MLX.withError, so a refused dispatch throws.
                    let parallel = try raw(input, strategy: .parallelThreshold)
                    let reference = try raw(input, strategy: .reference)
                    #expect(parallel.map { $0.sorted() } == reference.map { $0.sorted() })
                }
            }
        #endif

        @Test func cpuStreamHasOneImplementation() throws {
            try Device.withDefaultDevice(.gpu) { () throws -> Void in
                try Stream.withNewDefaultStream(device: .cpu) { () throws -> Void in
                    for input in Self.randomCases(seed: 3, count: 12, maximumPools: 2047) {
                        #expect(
                            try raw(input, strategy: .parallelThreshold)
                                == raw(input, strategy: .reference))
                    }
                }
            }
        }

        private static var devices: [DeviceType] {
            #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
                [.cpu, .gpu]
            #else
                [.cpu]
            #endif
        }

        @Test(arguments: devices)
        func selectTokensExecutionsAreIdentical(device: DeviceType) throws {
            try Device.withDefaultDevice(Device(device)) { () throws -> Void in
                var inputs = SelectionInputs(state: 41)
                for (history, queries, budget, dimension, coarse) in [
                    (37, 3, 8, 4, true), (130, 2, 16, 8, true), (2100, 1, 2048, 16, true),
                    (2100, 2, 2048, 16, false), (523, 4, 64, 8, false),
                ] {
                    let heads = 3
                    func array(_ shape: [Int], scale: Float = 1) -> MLXArray {
                        let count = shape.reduce(1, *)
                        // Coarse values make exact score ties (the fallback)
                        // common; continuous values exercise the unique path.
                        return MLXArray(
                            (0 ..< count).map { _ in
                                (coarse ? Float(inputs.int(5) - 2) : inputs.unit() * 4 - 2) * scale
                            }, shape)
                    }
                    let keyValidity = MLXArray(
                        (0 ..< history).map { $0 >= 2 && inputs.int(23) != 0 }, [1, history])
                    let queryValidity = MLXArray(
                        (0 ..< queries).map { $0 != 1 }, [1, queries])
                    let arguments = (
                        array([1, queries, heads, dimension]), array([1, history, dimension]),
                        array([1, history, dimension], scale: 0.5),
                        array([4, dimension], scale: 0.25),
                        array([1, queries, heads])
                    )
                    var results: [[Int32]] = []
                    for execution in [
                        GLM5NextSparseAttention.Execution.reference,
                        .init(selection: .parallelThreshold),
                        .init(deviceRanges: true),
                        .init(selection: .parallelThreshold, deviceRanges: true),
                    ] {
                        let selection = try GLM5NextSparseAttention.selectTokens(
                            indexQueries: arguments.0, indexKeys: arguments.1,
                            poolGates: arguments.2, poolPositionBias: arguments.3,
                            headWeights: arguments.4, keyValidity: keyValidity,
                            queryValidity: queryValidity, queryStart: history - queries,
                            tokenBudget: budget, execution: execution)
                        try fence([selection.indices])
                        #expect(selection.indices.dtype == .int32)
                        #expect(selection.indices.shape == [1, queries, budget + 3])
                        results.append(selection.indices.asArray(Int32.self))
                    }
                    #expect(results.allSatisfy { $0 == results[0] }, "history \(history)")
                }
            }
        }
    }

    /// Prepared only: never used to assert full-model throughput floors.
    @Suite(
        .serialized,
        .enabled(
            if:
                ProcessInfo.processInfo.environment["GLM53_RUN_SELECTION_BENCHMARK"] == "1"
                && ProcessInfo.processInfo.environment["GLM53_GPU_LANE_LOCKED"] == "1"))
    struct GLM5NextSparseSelectionBenchmark {
        @Test func interleavedReferenceAndParallelSelection() throws {
            try Device.withDefaultDevice(.gpu) { () throws -> Void in
                var inputs = SelectionInputs(state: 5)
                for pools in [128, 512, 2048, 8192, 32768] {
                    for tieHeavy in [false, true] {
                        let values = (0 ..< pools).map { _ in
                            tieHeavy ? Float(inputs.int(4)) : inputs.unit()
                        }
                        let scores = MLXArray(values, [1, pools])
                        let complete = MLXArray.ones([1, pools], dtype: .bool)
                        func run(_ strategy: GLM5NextSparseAttention.SelectionStrategy) throws
                            -> MLXArray
                        {
                            try GLM5NextSparseSelection.select(
                                scores: scores, completePools: complete, requestedCount: 512,
                                strategy: strategy)
                        }
                        func fence(_ array: MLXArray) throws {
                            try MLX.withError { error in
                                eval(array)
                                Stream.gpu.synchronize()
                                try error.check()
                            }
                        }
                        let reference = try run(.reference)
                        let parallel = try run(.parallelThreshold)
                        try fence(reference)
                        try fence(parallel)
                        guard
                            reference.asArray(Int32.self).sorted()
                                == parallel.asArray(Int32.self).sorted()
                        else { throw GLM5NextSparseSelection.Failure.invalidTensor }
                        for _ in 0 ..< 3 {
                            for strategy in [Strategy.reference, .parallelThreshold] {
                                try fence(run(strategy))
                            }
                        }
                        let clock = ContinuousClock()
                        for sample in 0 ..< 12 {
                            let order: [Strategy] =
                                sample.isMultiple(of: 2)
                                ? [.reference, .parallelThreshold]
                                : [.parallelThreshold, .reference]
                            for strategy in order {
                                Stream.gpu.synchronize()
                                let start = clock.now
                                try fence(run(strategy))
                                let elapsed = start.duration(to: clock.now).components
                                let record: [String: Any] = [
                                    "scope": "sparse-selection-only-not-full-model-throughput",
                                    "backend": strategy == .reference ? "reference" : "parallel",
                                    "sample": sample, "pools": pools, "count": 512,
                                    "tieHeavy": tieHeavy,
                                    "milliseconds": Double(elapsed.seconds) * 1000
                                        + Double(elapsed.attoseconds) / 1e15,
                                ]
                                print(
                                    String(
                                        decoding: try JSONSerialization.data(
                                            withJSONObject: record, options: [.sortedKeys]),
                                        as: UTF8.self))
                            }
                        }
                    }
                }
            }
        }

        private typealias Strategy = GLM5NextSparseAttention.SelectionStrategy
    }
}
