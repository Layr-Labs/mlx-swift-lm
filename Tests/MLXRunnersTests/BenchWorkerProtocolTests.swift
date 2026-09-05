// BenchWorkerProtocolTests.swift
//
// Engine Protocol v1 conformance for `BenchWorkerServer`, driven in process
// over a scripted runner. No weights, no Metal, no MLX allocator.
//
// The load-bearing case replays the SHARED fixture
// (`Resources/engine-wire-v1-adapter.ndjson`, pinned identically on the
// benchd side) and compares every response line BYTE FOR BYTE, key order
// included. A wire that merely parses the same is not the same wire: key
// order, omitted-vs-null, and the `0.0`/`10.0` rendering of a JSON number
// are all things one side can get wrong while still round-tripping.

import Foundation
import Testing

@testable import MLXRunners

@Suite("bench-worker Engine Protocol v1")
struct BenchWorkerProtocolTests {

    /// The fixture split into its request lines and its expected response
    /// lines. Line 0 is the unsolicited hello; the rest alternate.
    struct Fixture {
        let hello: String
        let requests: [String]
        let responses: [String]

        init() throws {
            let url = try #require(
                Bundle.module.url(
                    forResource: "engine-wire-v1-adapter", withExtension: "ndjson"))
            let lines = try String(contentsOf: url, encoding: .utf8)
                .split(separator: "\n", omittingEmptySubsequences: true)
                .map(String.init)
            hello = lines[0]
            var requests: [String] = []
            var responses: [String] = []
            for (index, line) in lines.dropFirst().enumerated() {
                if index % 2 == 0 { requests.append(line) } else { responses.append(line) }
            }
            self.requests = requests
            self.responses = responses
        }
    }

    private func makeServer(
        runner: any Runner, transport: ScriptedTransport, trusted: Bool = false
    ) -> BenchWorkerServer {
        BenchWorkerServer(
            runner: runner,
            transport: transport,
            trusted: trusted,
            build: "fixture",
            device: "mock",
            kvBytesCapacity: 1 << 20,
            maxDecodeTokens: 64,
            memory: FixtureMemoryReporter(),
            nonce: "fixturenonce")
    }

    // MARK: - Shared conformance fixture

    @Test("Every fixture response line is reproduced byte for byte")
    func fixtureBytes() async throws {
        let fixture = try Fixture()
        let transport = ScriptedTransport(lines: fixture.requests)
        let server = makeServer(runner: MockRunner(), transport: transport)
        await server.run()

        #expect(transport.written.count == fixture.responses.count + 1)

        // The hello, byte for byte like every other line — with ONE
        // substitution, which is a fixture defect rather than a difference of
        // opinion. The fixture spells `"backend":"mock"` while pinning the
        // section 11 manifest's digest, and that manifest declares
        // `"backend":"mlx"`. §6.1 derives `hello.backend` from
        // `manifest.backend`, so a worker cannot report "mock" and hash to
        // 474efd99… at the same time: the two bytes are the same fact told
        // twice, and only one of them can be right. Substituting here keeps
        // the assertion exact instead of loosening it; the fixture wants
        // `"backend":"mlx"`, or a mock-backend manifest and its own digest.
        let expectedHello = fixture.hello.replacingOccurrences(
            of: "\"backend\":\"mock\"", with: "\"backend\":\"mlx\"")
        #expect(transport.written[0] == expectedHello)
        #expect(
            transport.written[0].hasSuffix(
                ",\"runner\":{\"id\":\"layr/qwen4exp-125b-a6b\","
                    + "\"model_type\":\"qwen4_exp_text\",\"manifest_sha256\":"
                    + "\"474efd9965aef3453e1e8324e99f9711d8e44bb2dceb0366d9c14c7d8e9ecebe\","
                    + "\"build\":\"fixture\"}}"))

        for (index, expected) in fixture.responses.enumerated() {
            #expect(
                transport.written[index + 1] == expected,
                "response \(index + 1) diverged from the fixture")
        }
    }

    /// The fixture's own phase-close barriers, called out explicitly: the
    /// parent fails a run whose `completed_work` disagrees with the timed
    /// steps it issued.
    @Test("completed_work equals the timed steps issued in each phase")
    func completedWorkBarrier() async throws {
        let fixture = try Fixture()
        let transport = ScriptedTransport(lines: fixture.requests)
        let server = makeServer(runner: MockRunner(), transport: transport)
        await server.run()

        let decoder = JSONDecoder()
        let completed = try transport.written
            .map { try decoder.decode(WorkerResponse.self, from: Data($0.utf8)) }
            .compactMap(\.completedWork)
        // Phases in fixture order: prefill only (no timed steps);
        // decode_begin + decode_step; correctness only; correctness_begin +
        // correctness_step; free_decode_begin + three verify rounds.
        #expect(completed == [0, 2, 0, 2, 4])
    }

    // MARK: - Refusals

    @Test("free_decode_begin is refused when no free-run regime is declared")
    func refusesUnadvertisedFreeRun() async throws {
        let transport = ScriptedTransport(lines: [
            "{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens\":[51,52,53]}"
        ])
        let server = makeServer(runner: TeacherForcedOnlyMockRunner(), transport: transport)
        await server.run()

        let decoder = JSONDecoder()
        let response = try decoder.decode(
            WorkerResponse.self, from: Data(transport.written[1].utf8))
        #expect(response.id == 1)
        #expect(response.ok == false)
        #expect(response.nonce == "fixturenonce")
        #expect(response.error == "free_run_decode is not served by this adapter")
    }

    @Test("An unparseable line answers id -1 without ending the session")
    func refusesUnparseableLine() async throws {
        let transport = ScriptedTransport(lines: [
            "{ not json",
            "{\"id\":1,\"kind\":\"prefill\",\"prompt_tokens\":[11,12,13]}",
        ])
        let server = makeServer(runner: MockRunner(), transport: transport)
        await server.run()

        let decoder = JSONDecoder()
        let refusal = try decoder.decode(
            WorkerResponse.self, from: Data(transport.written[1].utf8))
        #expect(refusal.id == -1)
        #expect(refusal.ok == false)
        let served = try decoder.decode(
            WorkerResponse.self, from: Data(transport.written[2].utf8))
        #expect(served.id == 1)
        #expect(served.token == 100_003)
    }

    @Test("A spec mode the runner did not load is refused, never resolved")
    func refusesUnloadedMode() async throws {
        let transport = ScriptedTransport(lines: [
            "{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens\":[51],"
                + "\"spec\":{\"mode\":\"dflash\"}}"
        ])
        let server = makeServer(runner: MockRunner(), transport: transport)
        await server.run()

        let response = try JSONDecoder().decode(
            WorkerResponse.self, from: Data(transport.written[1].utf8))
        #expect(response.ok == false)
        #expect(response.error == "spec mode dflash was not advertised")
    }

    // MARK: - Hello derivation (§6.1)

    @Test("Hello derives every field from the manifest and the loaded state")
    func helloDerivation() async throws {
        let transport = ScriptedTransport(lines: [])
        let server = makeServer(runner: MockRunner(), transport: transport)
        let hello = server.hello()

        #expect(hello.id == 0)
        #expect(hello.ok)
        #expect(hello.nonce == "fixturenonce")
        #expect(hello.protocolVersion == 1)
        // From the manifest, never from a constant here.
        #expect(hello.backend == "mlx")
        #expect(hello.device == "mock")
        // Every declared regime is single-stream and free-run or
        // teacher-forced, so exactly one capability, and no max_batch_size.
        #expect(hello.capabilities == ["free_run_decode"])
        #expect(hello.maxBatchSize == nil)
        #expect(hello.specModes == ["serial", "mtp"])
        #expect(hello.headProvenance == nil)
        #expect(hello.runner?.id == "layr/qwen4exp-125b-a6b")
        #expect(hello.runner?.modelType == "qwen4_exp_text")
        #expect(hello.runner?.manifestSHA256 == MockRunner.manifest.sha256Digest())
        #expect(hello.runner?.build == "fixture")
    }

    @Test("The trusted build is the only one advertising cohort_reference_replay")
    func helloTrustedCapability() {
        let untrusted = makeServer(
            runner: MockRunner(), transport: ScriptedTransport(lines: []))
        #expect(untrusted.hello().capabilities?.contains("cohort_reference_replay") == false)

        let trusted = makeServer(
            runner: MockRunner(), transport: ScriptedTransport(lines: []), trusted: true)
        #expect(trusted.hello().capabilities?.contains("cohort_reference_replay") == true)
    }

    @Test("A cohort regime yields the batched capability and max_batch_size")
    func helloCohortDerivation() {
        // Gemma 4 declares an `upTo(8)` free-run regime; the derivation must
        // read the width off the manifest rather than a constant here.
        let widths = Gemma4TextRunner.manifest.regimes.map(\.batch.maxWidth)
        #expect(widths.max() == 8)
        #expect(
            Gemma4TextRunner.manifest.regimes.contains {
                $0.batch.maxWidth > 1 && $0.timing == .freeRun
            })
        #expect(Qwen3VLRunner.manifest.regimes.allSatisfy { $0.batch.maxWidth == 1 })
    }

    // MARK: - Wire round trip

    @Test("Every request kind decodes from its schema spelling")
    func requestKindsDecode() throws {
        let lines: [(String, WorkerRequest.Kind)] = [
            ("{\"id\":1,\"kind\":\"prefill\",\"prompt_tokens\":[1]}", .prefill),
            ("{\"id\":2,\"kind\":\"decode_begin\",\"seed_tokens\":[1]}", .decodeBegin),
            ("{\"id\":3,\"kind\":\"decode_step\",\"token\":1}", .decodeStep),
            ("{\"id\":4,\"kind\":\"correctness\",\"prompt_tokens\":[1],\"steps\":2}", .correctness),
            (
                "{\"id\":5,\"kind\":\"correctness_begin\",\"prompt_tokens\":[1]}",
                .correctnessBegin
            ),
            ("{\"id\":6,\"kind\":\"correctness_step\",\"token\":1}", .correctnessStep),
            ("{\"id\":7,\"kind\":\"phase_diagnostics\"}", .phaseDiagnostics),
            (
                "{\"id\":8,\"kind\":\"free_decode_begin\",\"seed_tokens_by_stream\":[[1],[2]],"
                    + "\"batch_size\":2,\"spec\":{\"mode\":\"mtp\",\"mtp\":{\"depth\":2}}}",
                .freeDecodeBegin
            ),
            (
                "{\"id\":9,\"kind\":\"free_decode_run\",\"count\":4,\"batch_size\":2}",
                .freeDecodeRun
            ),
            (
                "{\"id\":10,\"kind\":\"cohort_reference_replay\","
                    + "\"replay_seeds_by_stream\":[[1]],\"committed_by_stream\":[[2]],"
                    + "\"logit_top_k\":16,\"rel_envelope\":0.05,\"replay_width\":\"cohort\"}",
                .cohortReferenceReplay
            ),
        ]
        for (line, kind) in lines {
            let request = try JSONDecoder().decode(
                WorkerRequest.self, from: Data(line.utf8))
            #expect(request.kind == kind)
        }
    }

    @Test("Every response field round-trips through the wire writer")
    func responseRoundTrip() throws {
        var response = WorkerResponse(id: 7, ok: true, nonce: "n")
        response.token = 1
        response.topLogits = [CorrectnessTraceLogit(token: 1, logit: 10)]
        response.seedToken = 2
        response.tokens = [3, 4]
        response.expertStats = ExpertStreamingStats()
        response.peakRAMGB = 18.5
        response.protocolVersion = 1
        response.backend = "mlx"
        response.device = "m5"
        response.completedWork = 2
        response.cacheMemory = 0
        response.capabilities = ["free_run_decode"]
        response.acceptanceLengths = [1, 2]
        response.draftedTotal = 6
        response.acceptedTotal = 4
        response.committedTotal = 6
        response.effectiveSpec = SpecConfig(mode: "mtp", mtp: MtpSpec(depth: 2))
        response.specModes = ["serial", "mtp"]
        response.headProvenance = WireHeadProvenance(
            HeadProvenance(sha256: "abc", bytes: 12, fileCount: 1))
        response.mlxActiveMemoryBytes = 1
        response.mlxCacheMemoryBytes = 2
        response.mlxPeakMemoryBytes = 3
        response.topLogitMargin = 1.5
        response.expectedTokenLogit = 2.5
        response.expectedTokenRank = 0
        response.specDecoder = "mtp"
        response.maxBatchSize = 8
        response.seedTokenByStream = [1, 2]
        response.effectiveBatchSize = 2
        response.tokensByStream = [[1], [2]]
        response.naturalAcceptedByStream = [[1], [2]]
        response.rounds = 2
        response.activeStreamsByRound = [2, 1]
        response.depthClampReasons = ["batch_gate": 1]
        response.prefillNsByStream = [1, 2]
        response.decodeNsByStream = [3, 4]
        response.verifyReplayDisagreements = 0
        response.runner = RunnerIdentity(
            id: "layr/mock", modelType: "mock", manifestSHA256: "d", build: "b")

        let line = response.jsonLine()
        // `runner` LAST, per contract §6.0's appended-last convention.
        #expect(line.hasSuffix(
            ",\"runner\":{\"id\":\"layr/mock\",\"model_type\":\"mock\","
                + "\"manifest_sha256\":\"d\",\"build\":\"b\"}}"))

        let decoded = try JSONDecoder().decode(WorkerResponse.self, from: Data(line.utf8))
        #expect(decoded.id == 7)
        #expect(decoded.ok)
        #expect(decoded.tokensByStream == [[1], [2]])
        #expect(decoded.effectiveSpec == SpecConfig(mode: "mtp", mtp: MtpSpec(depth: 2)))
        #expect(decoded.depthClampReasons == ["batch_gate": 1])
        #expect(decoded.runner?.manifestSHA256 == "d")

        // Doubles keep their decimal point: a JSON `0` where the wire says
        // `0.0` is a different byte string, and both sides compare bytes.
        #expect(line.contains("\"expert_read_seconds\":0.0"))
        #expect(line.contains("\"logit\":10.0"))
        #expect(line.contains("\"peak_ram_gb\":18.5"))
    }

    @Test("An omitted optional is absent, never null")
    func omittedNotNull() {
        let line = WorkerResponse(id: 3, ok: true, nonce: "n").jsonLine()
        #expect(line == "{\"id\":3,\"nonce\":\"n\",\"ok\":true}")
    }
}
