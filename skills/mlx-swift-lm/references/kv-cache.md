# KV Cache System

## Overview

The KV (Key-Value) cache stores attention key and value tensors from previous tokens, enabling efficient autoregressive generation. Different cache types trade off between memory usage, context length, and performance.

| Class | Storage | Eviction | Use case |
|---|---|---|---|
| `StandardKVCache` | raw fp16 / bf16 | `.unbounded` or `.window(size:keep:)` | Default. Legacy `KVCacheSimple` (unbounded) and `RotatingKVCache` (windowed) collapsed into this single class. |
| `AffineQuantizedKVCache` | 4 / 6 / 8-bit affine group-quant | unbounded only | Memory-efficient. Self-transitions from raw → quantized at `startOffset` so prefill stays fast. Windowed-eviction requests fall back to raw `StandardKVCache(maxSize:)` per legacy `maybeQuantizeKVCache` swap behaviour. |
| `TurboQuantizedKVCache` | TurboQuant MSE codec, asymmetric K/V bits | unbounded **or** `.window(size:)` via `makeKVCache` (direct construction); `unbounded` only via `makeAttentionCache` (model-factory path, until [#185](https://github.com/ekryski/mlx-swift-lm/issues/185) is fixed) | Best memory ratio. `keyBits`=0 enables raw-key mode. Sliding-window via `rotatingMaxSize` / `rotatingIdx` machinery — verified working on Mistral 3 / Ministral 3 / Gemma 3. Gemma 4 produces incoherent output under windowed turbo (KV-shared + mixed sliding/full-attention layers); the model-factory path falls through to `StandardKVCache(maxSize:)` until that's investigated. `.keep` (attention-sink prefix) is not surfaced through the codec — windowed turbo treats the buffer as a flat rotating window. |
| `ArraysCache` | generic indexed `[MLXArray?]` slots | n/a | Building block for non-K/V caches. |
| `SSMStateCache` | conv state + recurrent state (subclass of `ArraysCache`) | n/a (SSM state is cumulative) | Mamba / GatedDeltaNet / hybrid linear-attention layers. Replaces legacy `MambaCache`. |
| `BatchedKVCache` | raw fp16 / bf16 across N streams | unbounded | Speculative decoding + multi-request servers. |
| `BatchedMambaCache` / `BatchedHybridCache` | batched SSM / hybrid | n/a | Batched analogues of `SSMStateCache` for hybrid models. |
| `PagedKVCache` | page-table-allocated raw K/V | unbounded | Paged attention; experimental. |
| `CacheList` | heterogeneous sub-caches | composite | Hybrid models (e.g. Qwen 3.5 GDN+Attention) that need multiple cache shapes per layer set. |

| Type | Use Case | Memory | Max Context |
|------|----------|--------|-------------|
| `KVCacheSimple` | Default, unbounded | Grows with context | Unlimited |
| `RotatingKVCache` | Long contexts | Fixed | `maxKVSize` |
| `QuantizedKVCache` | Memory-constrained | 4-8x less | Unlimited |
| `ChunkedKVCache` | Large prompt processing | Controlled | Chunked |
| `MambaCache` | Mamba/SSM models | Fixed state | N/A |

**File:** `Libraries/MLXLMCommon/KVCache.swift`

## Cache Types

### KVCacheSimple (Default)

Unbounded cache that grows with context length:

```swift
// Created automatically when no maxKVSize specified
let params = GenerateParameters()  // Uses KVCacheSimple
let cache = KVCacheSimple()

// Properties
cache.offset      // Current position in cache
cache.state       // [keys, values] for serialization
cache.isTrimmable // true
```

### RotatingKVCache (Sliding Window)

Fixed-size cache with sliding window attention:

```swift
// Enable via GenerateParameters
let params = GenerateParameters(maxKVSize: 4096)

### Runtime dispatch tag (`KVStorageKind`)

What the cache currently holds. Self-transitioning caches
(`AffineQuantizedKVCache`, `TurboQuantizedKVCache`) report their
*post-transition* state, so attention dispatch doesn't need `as?`
downcasts on concrete types.

```swift
public enum KVStorageKind: Sendable, Equatable {
    case raw
    case affineQuantized(bits: Int, groupSize: Int)
    case turboCompressed(keyBits: Int, valueBits: Int)
    case ssm
    case composite
}

cache.storageKind  // available on every KVCache
```

### User-facing string format (`KVCache.CompressionAlgorithm`)

The `GenerateParameters.compressionAlgorithm` parameter takes a
`KVCache.CompressionAlgorithm` (typealias for the top-level
`KVCacheCompressionAlgorithm`). It also has a string parser used by the
bench harness's `--kv` flag.

```swift
public enum KVCacheCompressionAlgorithm: Sendable, Equatable, CustomStringConvertible {
    case none
    case affine(bits: Int, groupSize: Int = 64)
    case turbo(
        keyBits: Int,
        valueBits: Int,
        skipBoundaryLayerCompression: Bool = true,
        boundaryLayersToSkip: Int = 2
    )
}

// Programmatic
let algo: KVCache.CompressionAlgorithm = .turbo(keyBits: 4, valueBits: 2)

// Or from a string (CLI / scheme):
let algo = KVCache.CompressionAlgorithm("turbo4v2")        // .turbo(4, 2)
let algo = KVCache.CompressionAlgorithm("turbo4")          // .turbo(4, 4)
let algo = KVCache.CompressionAlgorithm("turbo0v4")        // raw-key mode
let algo = KVCache.CompressionAlgorithm("affine4")         // .affine(4, 64)
let algo = KVCache.CompressionAlgorithm("affine4g32")      // .affine(4, 32)
let algo = KVCache.CompressionAlgorithm("affine8g32")      // .affine(8, 32)
let algo = KVCache.CompressionAlgorithm("none")            // .none
```

`description` round-trips: `algo.description == "turbo4v2"` etc.

## Factories

These are the call sites that ~14 model `newCache(parameters:)` factories
use. Don't hand-instantiate cache classes from outside the model
factories unless you're writing a custom cache strategy.

### `makeAttentionCache(parameters:maxSize:keep:)`

The 90% case for `newCache(parameters:)` — picks the right class based on
the parameters' `compressionAlgorithm`.

```swift
public func makeAttentionCache(
    parameters: GenerateParameters?,
    maxSize: Int? = nil,
    keep: Int = 0
) -> KVCache
```

Decision tree:
- `.affine(bits:groupSize:)` → `AffineQuantizedKVCache`. Window eviction is
  ignored (matches the legacy `maybeQuantizeKVCache` swap behaviour).
- `.turbo(...)` → caller's responsibility. Turbo construction needs
  per-model `headDim` for kernel JIT pre-warm + boundary-skip logic, so
  models that opt into turbo construct `TurboQuantizedKVCache` directly.
- `.none` / `nil` → `StandardKVCache(maxSize: maxSize, keep: keep)` if
  `maxSize` is set; else unbounded `StandardKVCache()`.

### `makeKVCache(scheme:eviction:)`

Single-cache factory composing the storage + eviction axes orthogonally.

```swift
public func makeKVCache(
    scheme: KVCache.CompressionAlgorithm = .none,
    eviction: KVEviction = .unbounded
) -> any KVCache
```

`.turbo(...)` + `.window(size:)` is supported — the codec's
`rotatingMaxSize` / `rotatingIdx` machinery wraps writes at `maxSize`
once the raw → compressed transition completes, and the SDPA path
honours windowed semantics for the mask. The `.keep` (attention-sink
prefix) parameter on `.window(...)` is not currently surfaced through
the TurboQuant codec; windowed turbo treats the buffer as a flat
rotating window. Use `.affine(bits:)` instead if you need the
attention-sink prefix.

### `turboBoundarySkipSet(attentionLayerIndices:algorithm:)`

For models that opt into `TurboQuant`, returns the set of attention-layer
indices that should stay uncompressed (first N / last N — most
PPL-sensitive).

```swift
public func turboBoundarySkipSet(
    attentionLayerIndices: [Int],
    algorithm: KVCache.CompressionAlgorithm?
) -> Set<Int>
```

Returns an empty set when the algorithm is `nil` / not turbo /
`skipBoundaryLayerCompression == false` / fewer than
`4 * boundaryLayersToSkip` attention layers (the floor exists so small
models like Qwen 3.5 0.8B don't end up with half their layers skipped).
Hybrid models like NemotronH thread Mamba / MLP / MoE layers around the
attention ones, so the caller computes `attentionLayerIndices` from its
own layer-type discovery.

Example pattern from `Qwen35TextModel.newCache`:

```swift
let layerIndices = (0..<args.hiddenLayers).filter {
    !linearLayerSet.contains($0)
}
let skipSet = turboBoundarySkipSet(
    attentionLayerIndices: layerIndices,
    algorithm: parameters?.compressionAlgorithm
)

// Behavior:
// - First 4 tokens always kept
// - After hitting maxSize, oldest tokens (except kept) are overwritten
// - Offset continues growing, but actual cache size is capped
```

### QuantizedKVCache

Memory-efficient cache using 4-bit or 8-bit quantization:

```swift
// Enable via GenerateParameters
let params = GenerateParameters(
    kvBits: 4,           // 4 or 8 bits
    kvGroupSize: 64,     // Quantization group size
    quantizedKVStart: 0  // Start quantizing after N tokens
)

// Or create directly
let cache = QuantizedKVCache(
    groupSize: 64,
    bits: 4,
    mode: .affine
)

// Use updateQuantized() instead of update()
let (qKeys, qValues) = cache.updateQuantized(keys: keys, values: values)
// qKeys = (weight, scales, biases?)
// qValues = (weight, scales, biases?)
```

### Dynamic Cache Quantization

Caches can be converted during generation:

```swift
// Simple cache converts to quantized after threshold
var cache: [KVCache] = model.newCache(parameters: nil)

// This happens automatically inside TokenIterator when:
// - kvBits is set
// - cache offset > quantizedKVStart
maybeQuantizeKVCache(
    cache: &cache,
    kvBits: 4,
    kvGroupSize: 64,
    quantizedKVStart: 0
)

// Manual conversion (KVCacheSimple only)
let simpleCache = KVCacheSimple()
// ... use cache ...
let quantizedCache = simpleCache.toQuantized(groupSize: 64, bits: 4)

// Convert back
let simpleAgain = quantizedCache.toUnquantized()
```

**Important:** `RotatingKVCache.toQuantized()` is **not implemented** and will `fatalError()`. The temporal ordering of a rotating cache makes quantization complex. If you need both sliding window and quantization, use `KVCacheSimple` with quantization and manage context length manually.

## Creating Caches

### Via Model

```swift
// Models create appropriate cache for their architecture
let cache = model.newCache(parameters: generateParameters)
```

### Via Utility Functions

```swift
// From model (recommended)
let cache = makePromptCache(model: model, parameters: params)

// With known layer count
let cache = makePromptCacheWithLayerCount(
    numLayers: 32,
    maxKVSize: 4096  // nil for unbounded
)
```

## Cache Operations

### Trimming

Remove tokens from the end of cache:

```swift
// Check if trimmable
if canTrimPromptCache(cache) {
    // Trim last 10 tokens
    let trimmed = trimPromptCache(cache, numTokens: 10)
}

// Direct trim
cache.first?.trim(10)
```

### Serialization

Save and load prompt cache for reuse:

```swift
// Save
try savePromptCache(
    url: fileURL,
    cache: cache,
    metadata: ["prompt": "My cached prompt"]
)

// Load
let (loadedCache, metadata) = try loadPromptCache(url: fileURL)
```

Cache files are `.safetensors` format with metadata.

## Attention Masks

Caches create appropriate attention masks:

```swift
// Modern API - cache creates its own mask
let mask = cache.makeMask(
    n: sequenceLength,
    windowSize: nil,  // Or specific window
    returnArray: false  // .causal vs .array
)

// Helper function
let mask = makeAttentionMask(
    n: n,
    cache: cache,
    windowSize: nil,
    returnArray: false
)

// Returns MLXFast.ScaledDotProductAttentionMaskMode:
// .none - no mask needed (single token)
// .causal - symbolic causal mask
// .array(MLXArray) - explicit mask array
```

## Memory Considerations

### Memory Usage by Cache Type

| Cache Type | Memory per Token | Example (8K context, 32 layers) |
|------------|------------------|--------------------------------|
| KVCacheSimple (fp16) | Full | ~512MB |
| RotatingKVCache | Fixed at maxKVSize | Capped at maxKVSize |
| QuantizedKVCache (4-bit) | ~1/4 of fp16 | ~128MB |

### Best Practices

```swift
// For chat applications with long history
let params = GenerateParameters(
    maxKVSize: 4096,  // Sliding window
    kvBits: 4         // Quantized
)

// For short interactions (no memory pressure)
let params = GenerateParameters()  // Simple unbounded cache

// Clear cache when conversation resets
await session.clear()
```

## Quantized Attention

Use with QuantizedKVCache for efficient attention:

```swift
if let qCache = cache as? QuantizedKVCacheProtocol {
    let (qKeys, qValues) = qCache.updateQuantized(keys: keys, values: values)

    let output = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: qKeys,
        quantizedValues: qValues,
        scale: scale,
        mask: .none,
        groupSize: qCache.groupSize,
        bits: qCache.bits
    )
}
```

## State Space Model Caches

### MambaCache

- **`.turbo(...)` + `.window(size:keep:)` ignores `keep`.** The codec's
  rotating buffer treats the window as flat — the attention-sink prefix
  parameter is not currently surfaced. Use `.affine(bits:)` on a
  separate `StandardKVCache` if you need the sink prefix.
- **`AffineQuantizedKVCache` ignores windowed eviction** even when passed
  through `makeAttentionCache(maxSize:)` — matches legacy
  `maybeQuantizeKVCache` behaviour. If you need both, manage context
  length manually.
- **TurboQuantizedKVCache values are not directly trimmable** —
  `isTrimmable` is `false`. The decode-side compressed store doesn't
  preserve a clean tail.
- **KV-sharing donors must not be quantized** — `isDonor` flagged caches
  return raw fp16 / bf16 K / V to shared layers. Self-transitioning
  caches respect `isDonor` and stay raw. If you set `isDonor` on a
  pre-constructed quantized cache, behaviour is undefined.
- **Self-transition timing** — `AffineQuantizedKVCache` stays raw until
  `startOffset` (default `0` for spec-006 callers), then transitions in
  place. Inspect `cache.storageKind` to see the *current* state.

```swift
let cache = MambaCache(leftPadding: nil)

// Access via subscript
cache[0] = convState
cache[1] = ssmState

// Create mask
let mask = cache.makeMask(N: sequenceLength)
```

### CacheList

Composite cache for hybrid architectures:

```swift
let cache = CacheList(kvCache, mambaCache)
let kv = cache[0] as! KVCacheSimple
let mamba = cache[1] as! MambaCache
```

## Deprecated Patterns

### Old createAttentionMask signature

```swift
// DEPRECATED: Array of caches
func createAttentionMask(h: MLXArray, cache: [KVCache]?, returnArray: Bool)

// USE INSTEAD: Single cache with windowSize
func createAttentionMask(
    h: MLXArray,
    cache: KVCache?,       // Single cache
    windowSize: Int?,      // Explicit window
    returnArray: Bool
) -> MLXFast.ScaledDotProductAttentionMaskMode

// Or use cache's method directly
cache.makeMask(n: n, windowSize: windowSize, returnArray: false)
```

### Direct cache.update() on QuantizedKVCache

```swift
// WRONG: QuantizedKVCache.update() will fatalError
let (k, v) = quantizedCache.update(keys: keys, values: values)  // Crashes!

// CORRECT: Use updateQuantized()
let (qKeys, qValues) = quantizedCache.updateQuantized(keys: keys, values: values)
```
