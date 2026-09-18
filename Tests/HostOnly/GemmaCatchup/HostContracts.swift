import Foundation

private final class Row: CBv2SequenceKV {}
private final class Tokens {
    let values: [Int]
    init(_ values: [Int]) { self.values = values }
}
private struct Work { let id: Int; let decode: Bool }

@main
struct HostContracts {
    static var checks = 0
    static func check(_ condition: @autoclosure () -> Bool, _ message: String) {
        precondition(condition(), message)
        checks += 1
    }

    static func main() {
        unchangedMembership()
        changedMembership()
        invalidationAndRelease()
        borrowedLayersAndIndependentBanks()
        rowRetirement()
        homogeneousSampling()
        mixedSampling()
        prefillAndEmptySampling()
        denseGateUpAdmission()
        denseGateUpNamespaces()
        prefillGlueAdmission()
        prefillGlueVectorAdmission()
        prefillNormalizationCarry()
        prefillPrefixAndScatterAdmission()
        deferredExpertConsumption()
        promptGeGLUAdmission()
        countingSortSpecification()
        decodeGlueAdmissionAndCoverage()
        routerFinalistsSpecification()
        scaledEmbeddingAdmission()
        qkvGeometryAndTailCoverage()
        b8ExpertAdmission()
        b8RoutingAdmissionAndLayout()
        unifiedPositionCycle()
        coordinatedPositionBinding()
        compactRootEpochAdmission()
        print("host-contracts: 26 scenarios, \(checks) assertions passed; MLX/device/model imports: none")
    }

    static func compactRootEpochAdmission() {
        let defaults = Gemma4CacheRootPolicy(environment: [:])
        check(!defaults.enabled(.decode) && !defaults.enabled(.mtpVerify), "root scopes default OFF")
        let decode = Gemma4CacheRootPolicy(environment: ["DARKBLOOM_GEMMA4_COMPACT_DECODE_ROOTS": "1"])
        check(decode.enabled(.decode) && !decode.enabled(.mtpVerify), "verify has independent admission")
        let verify = Gemma4CacheRootPolicy(environment: ["DARKBLOOM_GEMMA4_COMPACT_MTP_ROOTS": "1"])
        check(!verify.enabled(.decode) && verify.enabled(.mtpVerify), "decode has independent admission")
        for base in [UInt64(0), 99, UInt64.max - 1, UInt64.max] {
            for updates in [1, 2, 8] {
                for width in [1, 4, 16] {
                    func allowed(binding: UInt64 = 5, delta: Int? = nil, completed: Int? = nil, idle: Bool = true) -> Bool {
                        Gemma4CacheRootPolicy.validates(binding: binding,
                            updates: base &+ UInt64(delta ?? updates), previousBinding: 5,
                            previousUpdates: base, expectedUpdates: updates, expectedWidth: width,
                            completedWidth: completed ?? width, idle: idle)
                    }
                    check(allowed(), "completed exact update/width cycle including epoch wrap")
                    check(!allowed(binding: 6), "rebind invalidates capture")
                    check(!allowed(delta: 0), "stale output has no new cycle")
                    check(!allowed(delta: updates + 1), "unexpected additional forward declines")
                    check(!allowed(completed: width + 1), "wrong final width declines")
                    check(!allowed(idle: false), "partial cycle cannot compact")
                }
            }
        }
    }

    static func unifiedPositionCycle() {
        for layers in [1, 2, 30] {
            let cycle = Gemma4PositionCycle(layers: layers)
            for width in [1, 2, 8, 1024] {
                for _ in 0..<20 {
                    for layer in 0..<layers {
                        check(cycle.begin(layer: layer, count: width, inputMatches: true), "ordered uniform write accepted")
                        check(!cycle.begin(layer: layer, count: width, inputMatches: true), "nested write rejected")
                        check(cycle.finish(layer: layer, count: width) == (layer == layers - 1 ? .advanced : .more),
                              "exactly one shared advance per completed cycle")
                    }
                }
            }
            cycle.reset()
            check(!cycle.begin(layer: 0, count: 1, inputMatches: false), "foreign descriptor declines")
            check(!cycle.begin(layer: 0, count: 0, inputMatches: true), "zero width declines")
            check(!cycle.begin(layer: 0, count: Int(Int32.max) + 1, inputMatches: true), "non-int32 width declines")
            check(cycle.finish(layer: 0, count: 1) == .declined, "finish requires a pending write")
        }
        let cycle = Gemma4PositionCycle(layers: 2)
        check(!cycle.begin(layer: 1, count: 1, inputMatches: true), "skipped first owner declines")
        check(cycle.begin(layer: 0, count: 4, inputMatches: true), "first owner")
        check(cycle.finish(layer: 0, count: 4) == .more, "partial cycle")
        check(!cycle.begin(layer: 1, count: 1, inputMatches: true), "mixed widths decline")
        cycle.reset()
        check(cycle.begin(layer: 0, count: 1, inputMatches: true), "rebind resets pending width")
    }

    static func coordinatedPositionBinding() {
        final class Coordinator: CBv2PositionBindingCoordinator {
            var isActive = true
            var finishes = 0
            func accepts(_ caches: [any CBv2AttendingLayerCache]) -> Bool { caches.count == 2 }
            func finishBinding() { finishes += 1 }
        }
        let group = Coordinator()
        let caches = (0..<2).map { CBv2LayerCache(layerIndex: $0, kind: .init()) }
        for cache in caches { cache.positionBindingCoordinator = group }
        let bank = CBv2LayerCacheBank(caches: caches)
        let a = Row(), b = Row()
        _ = bank.layerCaches(rowStates: [[a, b]])
        check(caches.allSatisfy { $0.coordinatedBinds == 1 && $0.binds == 0 }, "coordinated bind bypasses per-cache rebuild")
        check(group.finishes == 1, "one coordinator completion after all rows bind")
        _ = bank.layerCaches(rowStates: [[a, b]])
        check(group.finishes == 1, "unchanged membership remains allocation-free fast path")
        bank.invalidateBoundComposition()
        _ = bank.layerCaches(rowStates: [[a, b]])
        check(group.finishes == 2, "out-of-band invalidation rebinds")
        bank.releaseBoundRows()
        check(caches.allSatisfy { $0.rows.isEmpty }, "coordinator release drops rows")
        group.isActive = false
        _ = bank.layerCaches(rowStates: [[a, b]])
        check(caches.allSatisfy { $0.binds == 1 }, "inactive coordinator preserves ordinary binding")
    }

    static func b8RoutingAdmissionAndLayout() {
        let root = "DARKBLOOM_GEMMA4_B8_EXPERT_EXECUTION"
        let rank = "DARKBLOOM_GEMMA4_B8_ROUTE_RANK"
        let prefix = "DARKBLOOM_GEMMA4_B8_ROUTE_PREFIX"
        let fold = "DARKBLOOM_GEMMA4_B8_ROUTE_FOLD"
        for environment in [[:], [root: "1"], [rank: "1", prefix: "1", fold: "1"]] {
            let policy = Gemma4B8RoutePolicy(environment: environment)
            check(!policy.rank && !policy.prefixBounds && !policy.fold, "no implicit routing admission")
        }
        let on = Gemma4B8RoutePolicy(environment: [root: "1", rank: "1", prefix: "1", fold: "1"])
        check(on.rank && on.prefixBounds && on.fold && !on.directInput, "independent route switches")
        for native in [false, true] {
            for prefix in [false, true] {
                let source = Gemma4B8RouteFoldSources.make(native: native, prefix: prefix)
                check(source.contains("fast::exp(ld[i] - maxval)"), "retain stock exponential")
                check(source.contains("const T w = T("), "retain BF16 softmax close before scaling")
                check(!source.contains("metal::precise::exp(score - max_score)"), "exclude historical softmax substitution")
                check(source.contains("if (fold_tid < 64u)"), "only complete rank SIMD groups proceed")
                check(source.contains("0x80000000u") == prefix, "tag emission follows producer variant")
            }
        }
        for count in [8, 32] {
            var owners = Set<Int>()
            for row in 0..<8 {
                for lane in 0..<count {
                    check(owners.insert(row * count + lane).inserted, "fold scratch rows do not alias")
                }
            }
            check(owners.count == 8 * count, "all shared slices covered")
        }
    }

    static func b8ExpertAdmission() {
        let key = "DARKBLOOM_GEMMA4_B8_EXPERT_EXECUTION"
        func accepts(_ policy: Gemma4B8ExpertPolicy, shape: [Int] = [8, 1, 2816],
                     eligible: Bool = true, bf16: Bool = true, prefill: Bool = false,
                     compiled: Bool = true) -> Bool {
            policy.admits(targetEligible: eligible, inputShape: shape, inputBF16: bf16,
                          scheduledPrefill: prefill, compiledActivation: compiled)
        }
        check(!accepts(.init(environment: [:])), "B8 experiment defaults OFF")
        let enabled = Gemma4B8ExpertPolicy(environment: [key: "1"])
        check(accepts(enabled), "exact B8 decode admission")
        check(!enabled.compiled && enabled.tightDown, "compiled graph is a separate opt-in")
        for shape in [[1, 1, 2816], [1, 8, 2816], [8, 2, 2816], [8, 1, 2815], [8, 2816]] {
            check(!accepts(enabled, shape: shape), "near geometry must fall back")
        }
        check(!accepts(enabled, eligible: false), "semantic target admission")
        check(!accepts(enabled, bf16: false), "BF16 closes required")
        check(!accepts(enabled, prefill: true), "prefill does not use decode contract")
        check(!accepts(enabled, compiled: false), "respect compiled activation opt-out")
        for cap in -2...8 {
            let policy = Gemma4B8ExpertPolicy(environment: [key: "1", "DARKBLOOM_GEMMA4_GU_RUN_CAP": String(cap)])
            check(policy.runCap == ([1, 2, 4].contains(cap) ? cap : 4), "power-of-two run cap")
        }
        let disabled = Gemma4B8ExpertPolicy(environment: [key: "1", "DARKBLOOM_GEMMA4_DECODE_FUSED_GEGLU": "0"])
        check(!accepts(disabled), "explicit GU opt-out")
        check(Gemma4B8ExpertPolicy.pairedStorageBytesPerLayer == 128 * 1408 * (352 * 4 + 44 * 2 * 2),
              "paired repack allocation estimate includes scales and biases")
    }

    static func unchangedMembership() {
        let cache = CBv2LayerCache(layerIndex: 0, kind: .init())
        let bank = CBv2LayerCacheBank(caches: [cache])
        let row = Row()
        for _ in 0..<1000 {
            let returned = bank.layerCaches(rowStates: [[row]])
            check(returned[0] === cache, "cache object must remain stable")
        }
        check(cache.binds == 1, "unchanged decode membership must not rebind")
        check(cache.rows[0] === row, "bank must retain the same owner")
    }

    static func changedMembership() {
        let cache = CBv2LayerCache(layerIndex: 0, kind: .init())
        let bank = CBv2LayerCacheBank(caches: [cache])
        let a = Row(), b = Row(), c = Row()
        let waves: [[Row]] = [[a,b], [b,a], [b], [b,c], [a,c,b]]
        for (index, rows) in waves.enumerated() {
            _ = bank.layerCaches(rowStates: rows.map { [$0] })
            check(cache.binds == index + 1, "join/shrink/reorder/replacement must rebind")
            check(cache.rows.count == rows.count, "row count must match")
            check(zip(cache.rows, rows).allSatisfy { $0 === $1 }, "row order must match")
        }
    }

    static func invalidationAndRelease() {
        let cache = CBv2LayerCache(layerIndex: 0, kind: .init())
        let bank = CBv2LayerCacheBank(caches: [cache])
        let row = Row()
        _ = bank.layerCaches(rowStates: [[row]])
        bank.invalidateBoundComposition()
        _ = bank.layerCaches(rowStates: [[row]])
        check(cache.binds == 2, "rollback invalidation must force same-row rebind")
        bank.releaseBoundRows()
        check(cache.binds == 3 && cache.rows.isEmpty, "release must unbind rows")
        bank.releaseBoundRows()
        check(cache.binds == 3, "repeated release must be a no-op")
        _ = bank.layerCaches(rowStates: [[row]])
        check(cache.binds == 4, "readmission after release must rebind")
        _ = bank.layerCaches(rowStates: [])
        _ = bank.layerCaches(rowStates: [])
        check(cache.binds == 5 && cache.rows.isEmpty, "stable empty membership must not rebind")
    }

    static func borrowedLayersAndIndependentBanks() {
        let borrower = CBv2LayerCache(layerIndex: 0, kind: .init(sharesKVWithLayer: 1))
        let owner = CBv2LayerCache(layerIndex: 1, kind: .init())
        let bank = CBv2LayerCacheBank(caches: [borrower, owner])
        let row = Row()
        _ = bank.layerCaches(rowStates: [[nil,row]])
        _ = bank.layerCaches(rowStates: [[nil,row]])
        check(borrower.binds == 0, "borrowed layer must remain rowless")
        check(owner.binds == 1 && owner.rows[0] === row, "first nonnil owner anchors identity")
        check(owner.retainsForBorrowers, "source must keep borrower chunk ownership")
        let other = CBv2LayerCache(layerIndex: 0, kind: .init())
        let otherBank = CBv2LayerCacheBank(caches: [other])
        _ = otherBank.layerCaches(rowStates: [[row]])
        check(other.binds == 1, "a second bank must not share memoized bindings")
        check(!other.retainsForBorrowers, "unborrowed source must not gain retention")
        bank.releaseBoundRows()
        check(other.rows.count == 1, "releasing one bank must not affect another")
    }

    static func rowRetirement() {
        let cache = CBv2LayerCache(layerIndex: 0, kind: .init())
        let bank = CBv2LayerCacheBank(caches: [cache])
        var owner: Row? = Row()
        weak var observer = owner
        _ = bank.layerCaches(rowStates: [[owner]])
        owner = nil
        check(observer != nil, "bound row must remain retained")
        bank.releaseBoundRows()
        check(observer == nil, "released bank fingerprint must not retain row")
    }

    static func homogeneousSampling() {
        for size in [1,2,4,8] {
            let work = (0..<size).reversed().map { Work(id: $0, decode: true) }
            let tokens = Tokens(work.map { $0.id + 100 })
            var slices = 0, joins = 0
            let result = CBv2MTPSampledTokenAssembly.assemble(
                work: work, decodeCount: size, decodeTokens: tokens, prefillTokens: [Int:Tokens](),
                id: { $0.id }, isDecode: { $0.decode },
                slice: { _, _ in slices += 1; return Tokens([]) },
                concatenate: { _ in joins += 1; return Tokens([]) })
            check(result.rows == work.map(\.id), "sample IDs must keep decode plan order")
            check(result.tokens === tokens, "all-decode must reuse the sampled object itself")
            check(slices == 0 && joins == 0, "all-decode must not build slice/concat operations")
        }
    }

    static func mixedSampling() {
        // Rows 8/9 represent unfinished-prefill/verify; they contribute no
        // ordinary sampled token. A completed prefill remains in plan order.
        let work = [Work(id:7,decode:true), Work(id:8,decode:false), Work(id:2,decode:false),
                    Work(id:9,decode:false), Work(id:5,decode:true)]
        var slices = 0, joins = 0
        let result = CBv2MTPSampledTokenAssembly.assemble(
            work: work, decodeCount: 2, decodeTokens: Tokens([70,50]),
            prefillTokens: [2:Tokens([20])], id: { $0.id }, isDecode: { $0.decode },
            slice: { tokens, range in slices += 1; return Tokens(Array(tokens.values[range])) },
            concatenate: { parts in joins += 1; return Tokens(parts.flatMap(\.values)) })
        check(result.rows == [7,2,5], "mixed samples must keep original scheduler order")
        check(result.tokens?.values == [70,20,50], "mixed token/index association must stay exact")
        check(slices == 2 && joins == 1, "mixed assembly must retain legacy operations")
    }

    static func prefillAndEmptySampling() {
        for size in 0...2 {
            let work = (0..<size).map { Work(id:$0,decode:false) }
            let values = Dictionary(uniqueKeysWithValues: work.map { ($0.id,Tokens([$0.id])) })
            var slices = 0, joins = 0
            let result = CBv2MTPSampledTokenAssembly.assemble(
                work: work, decodeCount: 0, decodeTokens: Optional<Tokens>.none,
                prefillTokens: values, id: { $0.id }, isDecode: { $0.decode },
                slice: { _, _ in slices += 1; return Tokens([]) },
                concatenate: { parts in joins += 1; return Tokens(parts.flatMap(\.values)) })
            check(result.rows == Array(0..<size), "prefill sample order must match")
            check(result.tokens?.values == (size == 0 ? nil : Array(0..<size)), "empty/prefill behavior")
            check(slices == 0 && joins == (size > 1 ? 1 : 0), "prefill assembly preserves legacy operations")
        }
    }

    static func denseGateUpAdmission() {
        let key = Gemma4DenseGateUpPolicy.environmentKey
        check(!Gemma4DenseGateUpPolicy.requested(environment: [:]), "candidate defaults off")
        for value in ["", "0", "true", "yes", " 1", "1 ", "2"] {
            check(!Gemma4DenseGateUpPolicy.requested(environment: [key: value]),
                  "only explicit exact opt-in admits the unqualified candidate")
        }
        check(Gemma4DenseGateUpPolicy.requested(environment: [key: "1"]), "exact opt-in")
        for enabled in [false, true] {
            for ndim in [2, 3, 4] {
                for batch in [0, 1, 2, 4, 8] {
                    for positions in [0, 1, 2, 5, 1024] {
                        for hidden in [1024, 2048, 2816, 4096] {
                            for bits in [4, 6, 8] {
                                for group in [32, 64, 128] {
                                    let expected = enabled && ndim == 3 && batch == 1
                                        && positions == 1 && hidden == 2816 && bits == 8 && group == 64
                                    check(Gemma4DenseGateUpPolicy.admits(enabled: enabled,
                                        ndim: ndim, batch: batch, positions: positions,
                                        hidden: hidden, bits: bits, groupSize: group) == expected,
                                        "B1 decode-only QAT geometry must fail closed")
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    static func denseGateUpNamespaces() {
        let suffixes = ["gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
                        "up_proj.weight", "up_proj.scales", "up_proj.biases"]
        for layer in [0, 15, 29] {
            let groups = ["model.layers.\(layer).mlp", "language_model.model.layers.\(layer).mlp"]
                .map { prefix in suffixes.map { "\(prefix).\($0)" } }
            for primary in 0...1 {
                for mask in 0..<64 {
                    let supplied = Set(groups[primary].enumerated().compactMap {
                        mask & (1 << $0.offset) != 0 ? $0.element : nil
                    })
                    let keys = Gemma4DenseGateUpPolicy.parameterKeys(layer: layer, contains: supplied.contains)
                    check(keys == (mask == 63 ? groups[primary] : nil), "partial six-tensor pair rejected")
                }
                for competingKey in groups[1 - primary] {
                    let supplied = Set(groups[primary] + [competingKey])
                    check(Gemma4DenseGateUpPolicy.parameterKeys(layer: layer, contains: supplied.contains) == nil,
                          "even a partial competing namespace must be rejected")
                }
            }
            let ambiguous = Set(groups.flatMap { $0 })
            check(Gemma4DenseGateUpPolicy.parameterKeys(layer: layer, contains: ambiguous.contains) == nil,
                  "two complete namespaces are ambiguous")
            let unrelated = Set(suffixes.map { "vision_tower.layers.\(layer).mlp.\($0)" }
                                + suffixes.map { "model.layers.\(layer).experts.\($0)" })
            check(Gemma4DenseGateUpPolicy.parameterKeys(layer: layer, contains: unrelated.contains) == nil,
                  "vision and expert tensors cannot bind the dense language pair")
        }
        check(Gemma4DenseGateUpPolicy.parameterKeys(layer: -1, contains: { _ in true }) == nil,
              "invalid layer index rejected")
    }

    static func prefillGlueAdmission() {
        let key = "DARKBLOOM_GEMMA4_PREFILL_GLUE"
        let off = Gemma4PrefillGluePolicy(environment: [:])
        check(off.context(scheduledPrefill: true, eligibleModel: true) == nil, "prefill defaults OFF")
        for raw in ["0", "true", "1 ", "", "yes"] {
            check(!Gemma4PrefillGluePolicy(environment: [key: raw]).enabled, "exact opt-in only")
        }
        let policy = Gemma4PrefillGluePolicy(environment: [key: "1"])
        check(!policy.vectorized && policy.chained, "vector loads require separate opt-in")
        check(policy.context(scheduledPrefill: false, eligibleModel: true) == nil,
              "rectangular MTP/legacy call is not scheduled prefill")
        check(policy.context(scheduledPrefill: true, eligibleModel: false) == nil,
              "off-topology models retain original path")
        let context = policy.context(scheduledPrefill: true, eligibleModel: true)!
        for batch in [0, 1, 2, 4, 8] {
            for length in [0, 1, 2, 3, 7, 64, 127, 512, 1024] {
                for width in [1024, 2048, 2816, 4096] {
                    for inputBF16 in [false, true] {
                        for weightBF16 in [false, true] {
                            for epsilon: Float in [0, 1e-6, 1e-5, .infinity, .nan] {
                                let rows = context.rows(shape: [batch, length, width], inputBF16: inputBF16,
                                    weightShape: [width], weightBF16: weightBF16, eps: epsilon)
                                let valid = batch > 0 && length >= 2 && width == 2816
                                    && inputBF16 && weightBF16 && epsilon == Float(1e-6)
                                check(rows == (valid ? batch * length : nil), "prefill geometry/rounding gate")
                            }
                        }
                    }
                }
            }
        }
        for shape in [[Int](), [2816], [2, 2816], [1, 2, 1, 2816],
                      [Int.max, 2, 2816], [1, Int(Int32.max) + 1, 2816], [-1, 2, 2816]] {
            check(context.rows(shape: shape, inputBF16: true, weightShape: [2816],
                weightBF16: true, eps: 1e-6) == nil, "rank/overflow must fail before grid lowering")
        }
        for weightShape in [[Int](), [1, 2816], [2815], [2817]] {
            check(context.rows(shape: [1, 2, 2816], inputBF16: true, weightShape: weightShape,
                weightBF16: true, eps: 1e-6) == nil, "weight shape must match exactly")
        }
        let unchained = Gemma4PrefillGluePolicy(environment: [key: "1",
            "DARKBLOOM_GEMMA4_PREFILL_GLUE_CHAIN": "0"])
        check(!unchained.context(scheduledPrefill: true, eligibleModel: true)!.chained,
              "chain can be isolated for native paired gates")
    }

    static func prefillGlueVectorAdmission() {
        for available in [false, true] {
            for contiguous in [false, true] {
                for offset: UInt in 0..<32 {
                    for allocated: UInt in [0, 65536] {
                        check(Gemma4PrefillGluePolicy.permitsVectorLoad(available: available,
                            rowContiguous: contiguous, byteOffset: offset, allocatedBytes: allocated)
                            == (available && contiguous && allocated > 0 && offset % 8 == 0),
                            "do not infer alignment from row-contiguity or force lazy evaluation")
                    }
                }
            }
        }
    }

    static func prefillNormalizationCarry() {
        var carry = Gemma4PrefillNormalizationCarry<Tokens>()
        let source = Tokens([1]), wrong = Tokens([1]), normalized = Tokens([2])
        check(carry.take(for: source) == nil, "new forward has no pending normalization")
        carry.publish(source: source, normalized: normalized)
        check(carry.take(for: source) === normalized, "same object is the only eligible consumer")
        check(carry.take(for: source) == nil, "normalization is consumed exactly once")
        carry.publish(source: source, normalized: normalized)
        check(carry.take(for: wrong) == nil, "equal values are not the same source")
        check(carry.take(for: source) == nil, "mismatch discards stale work")
        carry.publish(source: source, normalized: normalized)
        carry.publish(source: wrong, normalized: nil)
        check(carry.take(for: source) == nil, "a fallback layer clears pending work")
        var other = Gemma4PrefillNormalizationCarry<Tokens>()
        carry.publish(source: source, normalized: normalized)
        check(other.take(for: source) == nil, "forwards do not share handoff state")
        var transient: Tokens? = Tokens([3])
        weak var observer = transient
        other.publish(source: transient!, normalized: normalized)
        transient = nil
        check(observer != nil, "pending work retains its source until consumption")
        _ = other.take(for: wrong)
        check(observer == nil, "discarded work releases the retained source")
    }

    static func prefillPrefixAndScatterAdmission() {
        let root = "DARKBLOOM_GEMMA4_PREFILL_GLUE"
        let prefix = "DARKBLOOM_GEMMA4_PREFILL_BRANCH_PREFIX"
        let scatter = "DARKBLOOM_GEMMA4_PREFILL_PRENORM_GATHER"
        let defaultPolicy = Gemma4PrefillGluePolicy(environment: [root: "1"])
        check(!defaultPolicy.branchPrefix && !defaultPolicy.scatter, "new arms separately default OFF")
        check(Gemma4PrefillGluePolicy(environment: [prefix: "1", scatter: "1"])
            .context(scheduledPrefill: true, eligibleModel: true) == nil, "subflags cannot enable root")
        for raw in ["0", "true", " 1", "1 ", "", "yes"] {
            let policy = Gemma4PrefillGluePolicy(environment: [root: "1", prefix: raw, scatter: raw])
            check(!policy.branchPrefix && !policy.scatter, "subflags require exact opt-in")
        }
        let policy = Gemma4PrefillGluePolicy(environment: [root: "1", prefix: "1", scatter: "1"])
        let context = policy.context(scheduledPrefill: true, eligibleModel: true)!
        check(context.branchPrefix && context.scatter, "independent enabled arms")
        check(policy.context(scheduledPrefill: false, eligibleModel: true) == nil, "MTP rectangle excluded")
        for rows in [-1, 0, 1, 2, 7, 8, 9, 16, 1024, 8192, Int(Int32.max) / 8,
                     Int(Int32.max) / 8 + 1, Int.max] {
            for width in [1, 4, 8, 16] {
                for u32 in [false, true] {
                    let actual = context.scatterAssignments(rows: rows, indexShape: [rows, width], indicesUInt32: u32)
                    let admitted = rows >= 8 && rows <= Int(Int32.max) / 8 && width == 8 && u32
                    check(actual == (admitted ? rows * 8 : nil), "scatter assignment extent gate")
                }
            }
        }
        for shape in [[Int](), [8], [7, 8], [8, 1, 8]] {
            check(context.scatterAssignments(rows: 8, indexShape: shape, indicesUInt32: true) == nil,
                  "only exact row-slot plane admitted")
        }
        check(defaultPolicy.context(scheduledPrefill: true, eligibleModel: true)!
            .scatterAssignments(rows: 8, indexShape: [8, 8], indicesUInt32: true) == nil,
              "disabled scatter never builds a sort")
    }

    static func deferredExpertConsumption() {
        let root = "DARKBLOOM_GEMMA4_PREFILL_GLUE", tail = "DARKBLOOM_GEMMA4_PREFILL_EXPERT_TAIL_FUSION"
        check(!Gemma4PrefillGluePolicy(environment: [root: "1"]).expertTail, "expert tail separately defaults OFF")
        for raw in ["0", "true", " 1", "1 ", ""] {
            check(!Gemma4PrefillGluePolicy(environment: [root: "1", tail: raw]).expertTail,
                  "tail requires exact opt-in")
        }
        let ctx = Gemma4PrefillGluePolicy(environment: [root: "1", tail: "1"])
            .context(scheduledPrefill: true, eligibleModel: true)!
        check(ctx.expertAssignments(rows: 8, indexShape: [8, 8], indicesUInt32: true) == 64,
              "tail may own ordinary sort without enabling scatter")
        check(ctx.scatterAssignments(rows: 8, indexShape: [8, 8], indicesUInt32: true) == nil,
              "tail does not implicitly enable scatter")
        let unchained = Gemma4PrefillGluePolicy(environment: [root: "1", tail: "1",
            "DARKBLOOM_GEMMA4_PREFILL_GLUE_CHAIN": "0"])
            .context(scheduledPrefill: true, eligibleModel: true)!
        check(unchained.expertAssignments(rows: 8, indexShape: [8, 8], indicesUInt32: true) == nil,
              "unchained tail must not defer reduction")
        var resolving = Gemma4DeferredExpertState<Int, Tokens>(pending: 7)
        var calls = 0
        let first = resolving.resolve { value in calls += 1; return Tokens([value]) }
        let second = resolving.resolve { _ in calls += 1; return Tokens([]) }
        check(first === second && first?.values == [7] && calls == 1, "original reducer runs at most once")
        check(resolving.pending == nil && resolving.consumePending() == nil, "resolved result cannot be fused")
        var fused = Gemma4DeferredExpertState<Int, Tokens>(pending: 9)
        check(fused.pending == 9 && fused.pending == 9, "rejected eligibility peeks do not consume")
        check(fused.consumePending() == 9 && fused.consumePending() == nil, "fusion consumes once")
        check(fused.resolve { _ in calls += 1; return Tokens([]) } == nil && calls == 1,
              "consumed fusion never triggers a duplicate reduction")
        let existing = Tokens([10])
        var resolved = Gemma4DeferredExpertState<Int, Tokens>(resolved: existing)
        check(resolved.resolve { _ in Tokens([]) } === existing, "fallback result is retained unchanged")
        check(resolved.consumePending() == nil, "fallback is not a pending fusion")
        var owner: Tokens? = Tokens([11])
        weak var observer = owner
        var retirement = Gemma4DeferredExpertState<Tokens, Int>(pending: owner!)
        owner = nil
        check(observer != nil, "pending state retains exactly its input")
        _ = retirement.consumePending()
        check(observer == nil, "consumption drops pending owner retention")
    }

    static func promptGeGLUAdmission() {
        let root = "DARKBLOOM_GEMMA4_PREFILL_GLUE", arm = "DARKBLOOM_GEMMA4_PROMPT_GLUE"
        let off = Gemma4PrefillGluePolicy(environment: [root: "1"])
            .context(scheduledPrefill: true, eligibleModel: true)!
        check(off.gegluPlan(shape: [1024, 2112], inputBF16: true, compiledBaseline: true) == nil,
              "GeGLU separately defaults OFF")
        check(Gemma4PrefillGluePolicy(environment: [arm: "1"])
            .context(scheduledPrefill: true, eligibleModel: true) == nil, "GeGLU cannot enable root")
        for value in ["true", "0", "1 ", " 1", ""] {
            check(!Gemma4PrefillGluePolicy(environment: [root: "1", arm: value]).geglu,
                  "exact activation opt-in")
        }
        let ctx = Gemma4PrefillGluePolicy(environment: [root: "1", arm: "1"])
            .context(scheduledPrefill: true, eligibleModel: true)!
        for rows in [0, 1, 512, 1023, 1024, 8192, Int(Int32.max)] {
            for columns in [64, 704, 2112, 4096] {
                for bf16 in [false, true] {
                    for compiled in [false, true] {
                        for packed in [false, true] {
                            let shape = [rows, 1, packed ? columns * 2 : columns]
                            let plan = ctx.gegluPlan(shape: shape, inputBF16: bf16,
                                compiledBaseline: compiled, fusedHidden: packed ? columns : nil)
                            let eligible = rows >= 1024 && (columns == 704 || columns == 2112)
                                && bf16 && compiled && rows <= Int(Int32.max) / (columns / 4)
                            check((plan != nil) == eligible, "exact prompt geometry and compiled-policy gate")
                            if let plan {
                                check(plan.rows == rows && plan.columns == columns && plan.threads == rows * (columns / 4),
                                      "four elements per worker and checked grid")
                                check(plan.pitch == shape.last! && plan.upOffset == (packed ? columns : 0),
                                      "packed producer offset stays explicit")
                                check(plan.outputShape == [rows, 1, columns], "activation output preserves logical shape")
                            }
                        }
                    }
                }
            }
        }
        for shape in [[Int](), [2112], [-1, 2112], [Int.max, Int.max, 2112], [1024, 0]] {
            check(ctx.gegluPlan(shape: shape, inputBF16: true, compiledBaseline: true) == nil,
                  "invalid or overflowing shape never lowers to GPU")
        }
        check(ctx.gegluPlan(shape: [1024, 2112], inputBF16: true, compiledBaseline: true,
                            fusedHidden: Int.max) == nil, "packed width overflow rejected before multiplication")
        check(ctx.gegluPlan(shape: [1024, 1408], inputBF16: true, compiledBaseline: true,
                            fusedHidden: 2112) == nil, "mismatched packed producer rejected")
    }

    static func countingSortSpecification() {
        let root = "DARKBLOOM_GEMMA4_PREFILL_GLUE", flag = "DARKBLOOM_ROUTE_CSORT_PREFILL"
        let off = Gemma4PrefillGluePolicy(environment: [root: "1"])
            .context(scheduledPrefill: true, eligibleModel: true)!
        check(off.routePlan(scoreShape: [1, 1024, 128]) == nil, "counting sort defaults OFF")
        for bitset in [false, true] {
            for parallel in [false, true] {
                let ctx = Gemma4PrefillGluePolicy(environment: [root: "1", flag: "1",
                    "DARKBLOOM_ROUTE_CSORT_PREFILL_BITSET": bitset ? "1" : "0",
                    "DARKBLOOM_GEMMA4_PROMPT_GLUE2": parallel ? "1" : "0"])
                    .context(scheduledPrefill: true, eligibleModel: true)!
                for rows in [2, 8, 9, 31, 32, 33, 511, 512, 1024, 1025, 8192] {
                    let plan = ctx.routePlan(scoreShape: [1, rows, 128])
                    check((plan != nil) == (rows > 8), "64-key decode geometry stays out")
                    guard let plan else { continue }
                    check(plan.assignments == rows * 8 && plan.blocks == (rows * 8 + 255) / 256,
                          "histogram/scatter extent")
                    check(plan.bitset == (bitset && rows * 8 >= 4096), "bitset amortization threshold")
                    check(plan.parallelScan == (parallel && rows >= 1024 && plan.blocks % 8 == 0),
                          "eight-part scan only on complete block ranges")
                    for tied in [false, true] {
                        let keys = (0..<plan.assignments).map { UInt32(tied ? 127 : ($0 * 37 + $0 / 31) % 128) }
                        let result = RouteSortReference.run(keys, bitset: plan.bitset, parallelScan: plan.parallelScan)
                        let order = keys.indices.sorted { keys[$0] == keys[$1] ? $0 < $1 : keys[$0] < keys[$1] }
                        var inverse = Array(repeating: UInt32(0), count: order.count)
                        for (position, index) in order.enumerated() { inverse[index] = UInt32(position) }
                        check(result.rows == order.map { UInt32($0 / 8) }, "row gather order matches stable sort specification")
                        check(result.keys == order.map { keys[$0] }, "all tied/boundary keys keep stable order")
                        check(result.inverse == inverse, "inverse matches the same permutation")
                    }
                }
                for shape in [[1, 1, 128], [1, 1024, 127], [1, 1024, 129], [0, 2, 128],
                              [Int.max, 2, 128], [1, (1 << 25) + 1, 128], [1, 1, 2, 128]] {
                    check(ctx.routePlan(scoreShape: shape) == nil, "invalid key-domain/shape/grid bounds rejected")
                }
            }
        }
    }

    static func decodeGlueAdmissionAndCoverage() {
        let root = "DARKBLOOM_GEMMA4_FUSED_LAYER_GLUE"
        let drafter = "DARKBLOOM_GEMMA4_DRAFTER_NORM_RESIDUAL_FUSE"
        check(Gemma4DecodeGluePolicy(environment: [:]).context(target: true, validatedAssistant: false) == nil,
              "decode root defaults OFF")
        let target = Gemma4DecodeGluePolicy(environment: [root: "1"])
        check(target.context(target: false, validatedAssistant: true) == nil, "assistant is separately opt-in")
        let policy = Gemma4DecodeGluePolicy(environment: [root: "1", drafter: "1"])
        check(policy.context(target: false, validatedAssistant: false) == nil, "shape alone is no model role")
        for (axis, ctx) in [(2816, policy.context(target: true, validatedAssistant: false)!),
                            (1024, policy.context(target: false, validatedAssistant: true)!)] {
            for batch in [0, 1, 2, 4, 8, 9] {
                for width in [0, 1, 2, 4] {
                    let rows = ctx.rows(shape: [batch, width, axis], inputBF16: true,
                        weightShape: [axis], weightBF16: true, eps: 1e-6)
                    check(rows == (batch > 0 && width == 1 ? batch : nil), "decode-only row geometry")
                }
            }
            for batch in [1, 2, 4, 8, 9] {
                var seen = Set<Int>()
                for row in 0..<batch {
                    for lane in 0..<(axis / 4) {
                        for value in 0..<4 {
                            let address = row * axis + lane * 4 + value
                            check(address < batch * axis && seen.insert(address).inserted,
                                  "per-row kernel writes each address exactly once")
                        }
                    }
                }
                check(seen.count == batch * axis, "whole target/drafter plane covered")
            }
            for shape in [[1, 1, axis + 1], [1, axis], [Int.max, 1, axis],
                          [Int(UInt32.max) / axis + 1, 1, axis]] {
                check(ctx.rows(shape: shape, inputBF16: true, weightShape: [axis],
                    weightBF16: true, eps: 1e-6) == nil, "invalid rank/extent/index arithmetic rejected")
            }
            check(ctx.rows(shape: [1, 1, axis], inputBF16: false, weightShape: [axis],
                weightBF16: true, eps: 1e-6) == nil, "decode dtype must match")
            check(ctx.rows(shape: [1, 1, axis], inputBF16: true, weightShape: [axis],
                weightBF16: true, eps: 1e-5) == nil, "decode epsilon must match")
        }
        let assistant = policy.context(target: false, validatedAssistant: true)!
        check(!assistant.paired && !assistant.chained, "assistant only receives norm/residual fusion")
    }

    static func routerFinalistsSpecification() {
        let key = "DARKBLOOM_GEMMA4_ROUTER_FINALISTS32"
        let off = Gemma4RouterFinalistsPolicy(environment: [:])
        check(off.plan(targetEligible: true, shape: [1, 1, 128], scoresBF16: true, scaleBF16: true,
            scaleShape: [128], topK: 8, scheduledPrefill: false) == nil, "finalists root defaults OFF")
        let policy = Gemma4RouterFinalistsPolicy(environment: [key: "1",
            "DARKBLOOM_GEMMA4_ROUTER_FINALISTS32_PREFILL": "1", "DARKBLOOM_GEMMA4_ROUTER_WEIGHTS32_PREFILL": "1"])
        for batch in [0, 1, 2, 4, 8, Int.max] {
            for width in [1, 2, 1024] {
                for prefill in [false, true] {
                    let plan = policy.plan(targetEligible: true, shape: [batch, width, 128], scoresBF16: true,
                        scaleBF16: true, scaleShape: [128], topK: 8, scheduledPrefill: prefill)
                    let valid = batch > 0 && batch <= Int(Int32.max) / 128 / width && (width == 1 || prefill)
                    check((plan != nil) == valid, "phase and grid bounds")
                    if let plan { check(plan.fusedWeights == (width > 1) && plan.rows == batch * width,
                                        "decode keeps ordinary weight chain") }
                }
            }
        }
        let special: [UInt16] = [0, 0x8000, 1, 0x007f, 0x0080, 0x8001, 0x807f, 0x8080,
                                 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc0, 0xffc0]
        for flush in [false, true] {
            for seed in 0..<96 {
                let bits = (0..<128).map { index -> UInt16 in
                    seed < 14 ? special[(index + seed) % special.count]
                        : UInt16(truncatingIfNeeded: index * 521 + seed * 997)
                }
                let packed = bits.enumerated().map { (UInt32($0.element) << 7) | UInt32($0.offset) }
                let expected = packed.sorted { FinalistsReference.before($0, $1, flush: flush) }
                    .suffix(8).map { Int($0 & 127) }
                check(FinalistsReference.select(bits, flush: flush) == expected,
                      "two-stage finalists network preserves stable index tie order")
                let keys = bits.enumerated().map { FinalistsReference.orderedKey($0.element, expert: $0.offset, flush: flush) }
                check(keys.allSatisfy { $0 > 0 } && Set(keys).count == 128, "max8 removal sentinel is distinct")
                check(keys.sorted().suffix(8).map { Int($0 & 127) } == expected,
                      "native-key ordinal preserves comparator semantics under either subnormal mode")
            }
        }
    }

    static func scaledEmbeddingAdmission() {
        let key = "DARKBLOOM_GEMMA4_SCALED_EMBEDDING", decode = "DARKBLOOM_GEMMA4_SCALED_EMBEDDING_DECODE"
        let off = Gemma4ScaledEmbeddingPolicy(environment: [:])
        check(off.plan(targetEligible: true, tokenShape: [1, 8], tokensInt32: true, vocab: 32,
            hidden: 2816, bits: 4, groupSize: 64) == nil, "embedding defaults OFF")
        for decodeOn in [false, true] {
            let policy = Gemma4ScaledEmbeddingPolicy(environment: [key: "1", decode: decodeOn ? "1" : "0"])
            for shape in [[1, 1], [1, 2], [2, 4], [8, 1024], [0, 1], [Int.max, 2], [2, 1, 1]] {
                let plan = policy.plan(targetEligible: true, tokenShape: shape, tokensInt32: true,
                    vocab: 32, hidden: 2816, bits: 4, groupSize: 64)
                let valid = shape.count == 2 && shape[0] > 0 && shape[0] <= Int(Int32.max) / shape[1]
                    && (shape[1] > 1 || decodeOn)
                check((plan != nil) == valid, "explicit decode and checked embedding grid")
                if let plan { check(plan.rows == shape[0] * shape[1] && plan.outputShape == shape + [2816], "embedding shape") }
            }
            for bits in [2, 4, 8] {
                for group in [32, 64, 128] {
                    check((policy.plan(targetEligible: true, tokenShape: [1, 8], tokensInt32: true,
                        vocab: 32, hidden: 2816, bits: bits, groupSize: group) != nil) == (bits == 4 && group == 64),
                        "quantization is admission, never conversion")
                }
            }
            check(policy.plan(targetEligible: false, tokenShape: [1, 8], tokensInt32: true,
                vocab: 32, hidden: 2816, bits: 4, groupSize: 64) == nil, "other models excluded")
            check(policy.plan(targetEligible: true, tokenShape: [1, 8], tokensInt32: false,
                vocab: 32, hidden: 2816, bits: 4, groupSize: 64) == nil, "token dtype preserved")
        }
    }

    static func qkvGeometryAndTailCoverage() {
        let policy = Gemma4QKVNormPolicy(environment: ["DARKBLOOM_GEMMA4_QKV_NORM": "1",
            "DARKBLOOM_GEMMA4_QKV_NORM_PREFILL": "1", "DARKBLOOM_GEMMA4_QKV_NORM_ROPE": "1"])
        for batch in [1, 2, 4, 8] {
            for (dimension, heads, shared) in [(256, 8, false), (512, 2, true)] {
                for (lq, lk) in [(1, 1), (2, 2), (1024, 1024), (1025, 1025), (1, 1025)] {
                    let plan = policy.plan(targetEligible: true, q: [batch, lq, 16, dimension],
                        k: [batch, lk, heads, dimension], v: [batch, lk, heads, dimension], bf16: true,
                        weightsMatch: true, eps: 1e-6, keyValueShared: shared, scheduledPrefill: true)
                    let admitted = lq == 1 && lk == 1 || batch * max(lq, lk) >= 1024
                    check((plan != nil) == admitted, "QKV geometry and prefill floor")
                    guard let plan else { continue }
                    check(plan.grid % plan.threads == 0 && plan.totalRows == plan.queryRows + plan.keyRows * (shared ? 1 : 2),
                          "row banks and threadgroup extent")
                    if plan.kind != .decode {
                        var qSeen = Set<Int>(), kSeen = Set<Int>(), vSeen = Set<Int>()
                        let launchedRows = plan.grid / (dimension / 4)
                        for row in 0..<launchedRows {
                            guard row < plan.totalRows else { continue } // masked slots still meet all barriers
                            let query = row < plan.queryRows
                            let key = row >= plan.queryRows && row < plan.queryRows + plan.keyRows
                            let local = query ? row : (key ? row - plan.queryRows : row - plan.queryRows - plan.keyRows)
                            let length = query ? lq : lk, bankHeads = query ? 16 : heads
                            let b = local / (length * bankHeads), rem = local % (length * bankHeads)
                            let position = rem / bankHeads, head = rem % bankHeads
                            let outputRow = (b * bankHeads + head) * length + position
                            if query { check(qSeen.insert(outputRow).inserted, "Q head-major ownership") }
                            else if key {
                                check(kSeen.insert(outputRow).inserted, "K head-major ownership")
                                if shared { check(vSeen.insert(outputRow).inserted, "K=V uses same raw row") }
                            } else { check(vSeen.insert(outputRow).inserted, "V head-major ownership") }
                        }
                        check(qSeen.count == batch * lq * 16 && kSeen.count == batch * lk * heads
                            && vSeen.count == batch * lk * heads, "complete output planes, including partial final group")
                    }
                }
            }
        }
        check(policy.plan(targetEligible: true, q: [8, 1024, 16, 512], k: [8, 1024, 2, 512],
            v: [8, 1024, 2, 512], bf16: true, weightsMatch: true, eps: 1e-6,
            keyValueShared: true, scheduledPrefill: false) == nil, "MTP rectangle cannot enter prefill kernel")
        check(policy.plan(targetEligible: true, q: [Int.max, 1, 16, 512], k: [Int.max, 1, 2, 512],
            v: [Int.max, 1, 2, 512], bf16: true, weightsMatch: true, eps: 1e-6,
            keyValueShared: true, scheduledPrefill: false) == nil, "overflow fails before lowering")
    }
}
