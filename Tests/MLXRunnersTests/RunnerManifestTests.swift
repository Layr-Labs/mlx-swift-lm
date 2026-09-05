// RunnerManifestTests.swift
//
// Manifest digest gates (Darkbloom runner contract §6.0).
//
// The canonical JSON bytes are PINNED here, spelled out in full, next to
// their sha256. A manifest edit therefore shows up as a diff of the exact
// bytes benchd will hash, not as an opaque hash change nobody can read — and
// a re-ordering of the declared fields cannot hide behind a stable digest,
// which is the reason the encoder walks the declared order instead of
// sorting keys.
//
// Model-free by construction: nothing here loads weights or touches Metal.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXRunners

@Suite("Runner manifest digests")
struct RunnerManifestTests {

    private func canonical(_ manifest: RunnerManifest) -> String {
        String(decoding: manifest.canonicalJSON(), as: UTF8.self)
    }

    /// The Qwen 3.8 Flash-Next runner's own manifest against the contract's
    /// §11 declaration. The digest is the cross-repo test vector: benchd pins
    /// the same string, so these bytes are the interface, not an internal
    /// detail.
    @Test("Section 11 manifest reproduces the cross-repo test vector")
    func sectionElevenVector() {
        let manifest = Qwen4ExpRunner.manifest

        #expect(
            canonical(manifest) == """
                {"schemaVersion":1,"runnerID":"layr/qwen4exp-125b-a6b","modelTypes":\
                ["qwen4_exp","qwen4_exp_text"],"backend":"mlx","engine":\
                {"supportsPrefixReuse":false,"supportsPagedKV":false,\
                "supportsCompiledDecode":false,"supportsPackedPrefill":false,\
                "supportsMTP":true,"supportsCompactRecurrentMTPReplay":false},\
                "kvBackends":["contiguous"],"decoders":[{"mode":"serial",\
                "drafter":"none","state":"stateless","depth":null},{"mode":"mtp",\
                "drafter":"embeddedHead","state":"requestStateful","depth":[1,3]}],\
                "regimes":[{"batch":"single","timing":"freeRun","perStreamTiming":false},\
                {"batch":"single","timing":"teacherForced","perStreamTiming":false}],\
                "multimodal":false,"recurrentLayers":true,"requiresKeepMask":true}
                """)
        #expect(
            manifest.sha256Digest()
                == "474efd9965aef3453e1e8324e99f9711d8e44bb2dceb0366d9c14c7d8e9ecebe")
    }

    @Test("Gemma 4 text manifest")
    func gemma4TextManifest() {
        #expect(
            canonical(Gemma4TextRunner.manifest) == """
                {"schemaVersion":1,"runnerID":"layr/gemma4-text","modelTypes":\
                ["gemma4","gemma4_text"],"backend":"mlx","engine":\
                {"supportsPrefixReuse":true,"supportsPagedKV":true,\
                "supportsCompiledDecode":true,"supportsPackedPrefill":true,\
                "supportsMTP":true,"supportsCompactRecurrentMTPReplay":false},\
                "kvBackends":["contiguous","paged"],"decoders":[{"mode":"serial",\
                "drafter":"none","state":"stateless","depth":null},{"mode":"mtp",\
                "drafter":"assistantCheckpoint","state":"stateless","depth":[1,7]}],\
                "regimes":[{"batch":"single","timing":"freeRun","perStreamTiming":false},\
                {"batch":{"upTo":8},"timing":"freeRun","perStreamTiming":false},\
                {"batch":"single","timing":"teacherForced","perStreamTiming":false}],\
                "multimodal":false,"recurrentLayers":false,"requiresKeepMask":false}
                """)
        #expect(
            Gemma4TextRunner.manifest.sha256Digest()
                == "e1890e74e9cefc901393f3826b26efeb9165d5a21c3505daea566487bf40b63d")
    }

    @Test("GPT-OSS manifest")
    func gptossManifest() {
        #expect(
            canonical(GPTOSSRunner.manifest) == """
                {"schemaVersion":1,"runnerID":"layr/gptoss","modelTypes":["gpt_oss"],\
                "backend":"mlx","engine":{"supportsPrefixReuse":true,\
                "supportsPagedKV":true,"supportsCompiledDecode":true,\
                "supportsPackedPrefill":true,"supportsMTP":false,\
                "supportsCompactRecurrentMTPReplay":false},\
                "kvBackends":["contiguous","paged"],"decoders":[{"mode":"serial",\
                "drafter":"none","state":"stateless","depth":null}],\
                "regimes":[{"batch":"single","timing":"freeRun","perStreamTiming":false},\
                {"batch":{"upTo":8},"timing":"freeRun","perStreamTiming":false},\
                {"batch":"single","timing":"teacherForced","perStreamTiming":false}],\
                "multimodal":false,"recurrentLayers":false,"requiresKeepMask":false}
                """)
        #expect(
            GPTOSSRunner.manifest.sha256Digest()
                == "2afbe23fc67ea9f2e9fd017098bb8b2877166190c9d56966dbeaa05656fcf8a2")
    }

    @Test("Qwen 3.5 manifest")
    func qwen35Manifest() {
        #expect(
            canonical(Qwen35Runner.manifest) == """
                {"schemaVersion":1,"runnerID":"layr/qwen35","modelTypes":\
                ["qwen3_5","qwen3_5_moe","qwen3_5_text"],"backend":"mlx","engine":\
                {"supportsPrefixReuse":false,"supportsPagedKV":false,\
                "supportsCompiledDecode":false,"supportsPackedPrefill":true,\
                "supportsMTP":true,"supportsCompactRecurrentMTPReplay":true},\
                "kvBackends":["contiguous"],"decoders":[{"mode":"serial",\
                "drafter":"none","state":"stateless","depth":null},{"mode":"mtp",\
                "drafter":"embeddedHead","state":"requestStateful","depth":[1,7]}],\
                "regimes":[{"batch":"single","timing":"freeRun","perStreamTiming":false},\
                {"batch":{"upTo":8},"timing":"freeRun","perStreamTiming":false},\
                {"batch":"single","timing":"teacherForced","perStreamTiming":false}],\
                "multimodal":false,"recurrentLayers":true,"requiresKeepMask":false}
                """)
        #expect(
            Qwen35Runner.manifest.sha256Digest()
                == "b8c3625c218dcf592518ebca1e64f3b41307b403753871aa7978dd5d0a12c896")
    }

    @Test("Qwen3-VL manifest")
    func qwen3vlManifest() {
        #expect(
            canonical(Qwen3VLRunner.manifest) == """
                {"schemaVersion":1,"runnerID":"layr/qwen3vl","modelTypes":\
                ["qwen3_vl","qwen3_vl_moe"],"backend":"mlx","engine":\
                {"supportsPrefixReuse":false,"supportsPagedKV":false,\
                "supportsCompiledDecode":false,"supportsPackedPrefill":false,\
                "supportsMTP":false,"supportsCompactRecurrentMTPReplay":false},\
                "kvBackends":["contiguous"],"decoders":[{"mode":"serial",\
                "drafter":"none","state":"stateless","depth":null}],\
                "regimes":[{"batch":"single","timing":"freeRun","perStreamTiming":false},\
                {"batch":"single","timing":"teacherForced","perStreamTiming":false}],\
                "multimodal":true,"recurrentLayers":false,"requiresKeepMask":false}
                """)
        #expect(
            Qwen3VLRunner.manifest.sha256Digest()
                == "a9145332bb9a78961831bc365af055348a8cedc45f8556ad58573ec32d821d3c")
    }

    /// Round trip through `Codable` in the §6.0 encodings: `batch` as the
    /// string `"single"` or the object `{"upTo": n}`, `depth` as `[lo, hi]`
    /// or `null`. The synthesized enum/range forms would decode to the same
    /// Swift values and to DIFFERENT bytes, so the assertion is on the JSON.
    @Test("Manifest Codable uses the pinned wire encodings")
    func codableEncodings() throws {
        let manifest = Gemma4TextRunner.manifest
        let encoded = try JSONEncoder().encode(manifest)
        let text = String(decoding: encoded, as: UTF8.self)
        #expect(text.contains("\"batch\":{\"upTo\":8}"))
        #expect(text.contains("\"batch\":\"single\""))
        #expect(text.contains("\"depth\":[1,7]"))
        #expect(text.contains("\"depth\":null"))

        let decoded = try JSONDecoder().decode(RunnerManifest.self, from: encoded)
        #expect(decoded == manifest)
        #expect(decoded.sha256Digest() == manifest.sha256Digest())
    }
}
