import Foundation
import MLX
import MLXLLM
import MLXNN
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif

/// Mock-tokenizer routing tests are not real tokenizer/Jinja qualification.
/// Native tests use the existing generated BF16 payload and explicit lane gate.

private func withMiMoConstructionScope<Value>(
    _ body: (NativeConstructionScope) throws -> Value
) rethrows -> Value {
    let work = NativeConstructionScope()
    defer {
        // Unexpected failed completion is restart-only, including in this
        // dedicated native test process. Never deallocate its sole SDK owner.
        if work.snapshot.isRetainedFault { _ = Unmanaged.passRetained(work) }
    }
    return try body(work)
}

final class MiMoV26FactoryTests: XCTestCase {
    private enum ProbeError: Error { case tokenizer, callback, revoked }
    private final class Permit: MiMoV26SerialLoadReservation, @unchecked Sendable {
        let request: MiMoV26SerialLoadRequest
        let reservedLoadBytes: UInt64
        private let lock = NSLock()
        private var revoked = false
        private var count = 0
        var validations: Int {
            lock.lock()
            defer { lock.unlock() }
            return count
        }
        init(_ request: MiMoV26SerialLoadRequest) {
            self.request = request
            reservedLoadBytes = request.requiredLoadBytes
        }
        func revoke() {
            lock.lock()
            revoked = true
            lock.unlock()
        }
        func validateActive(progress: MiMoV26SerialLoadProgress) throws {
            lock.lock()
            defer { lock.unlock() }
            count += 1
            if revoked { throw ProbeError.revoked }
        }
    }
    private final class MockTokenizer: Tokenizer, @unchecked Sendable {
        struct Call: Sendable {
            let messages: [Message]
            let template: String
            let tools: [[String: any Sendable]]?
            let context: [String: any Sendable]?
        }
        private let lock = NSLock()
        private var recorded: [Call] = []
        private var directories: [URL] = []
        var calls: [Call] {
            lock.lock()
            defer { lock.unlock() }
            return recorded
        }
        var loadedDirectories: [URL] {
            lock.lock()
            defer { lock.unlock() }
            return directories
        }
        func loaded(from directory: URL) {
            lock.lock()
            directories.append(directory)
            lock.unlock()
        }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [6] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map(String.init).joined(separator: " ")
        }
        func convertTokenToId(_ token: String) -> Int? {
            ["<|im_start|>": 1, "<|im_end|>": 3, "<think>": 4, "</think>": 5, "<|extra_eos|>": 7][
                token]
        }
        func convertIdToToken(_ id: Int) -> String? { id == 3 ? "<|im_end|>" : nil }
        var bosToken: String? { nil }
        var eosToken: String? { "<|im_end|>" }
        var unknownToken: String? { nil }
        func applyChatTemplate(
            messages: [Message], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            throw TokenizerError.missingChatTemplate
        }
        func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
            try applyChatTemplate(
                messages: messages, chatTemplate: chatTemplate, tools: nil, additionalContext: nil)
        }
        func applyChatTemplate(
            messages: [Message], chatTemplate: String, tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            lock.lock()
            recorded.append(
                .init(
                    messages: messages, template: chatTemplate,
                    tools: tools, context: additionalContext))
            lock.unlock()
            return [1, 2]
                + ((additionalContext?["enable_thinking"] as? Bool) == false ? [4, 5] : []) + [6]
        }
    }
    private struct Loader: TokenizerLoader {
        let tokenizer: MockTokenizer
        var fail = false
        var mutateTemplate: URL?
        func load(from directory: URL) async throws -> any Tokenizer {
            tokenizer.loaded(from: directory)
            if fail { throw ProbeError.tokenizer }
            if let mutateTemplate { try Data("changed template".utf8).write(to: mutateTemplate) }
            return tokenizer
        }
    }
    private let literal =
        "{{ messages }}{% if enable_thinking is false %}<think></think>{% endif %}\n"

    private func fixture(allEOS: Bool = false) throws -> (URL, MiMoV26SerialLoadSession) {
        guard let path = ProcessInfo.processInfo.environment["MIMO_V26_SERIAL_LOAD_FIXTURES"] else {
            throw XCTSkip("Requires existing generated MIMO_V26_SERIAL_LOAD_FIXTURES")
        }
        let fixtures = URL(fileURLWithPath: path)
        let root = fixtures.appendingPathComponent("factory-work-" + UUID().uuidString)
        try FileManager.default.copyItem(at: fixtures.appendingPathComponent("tiny-bf16"), to: root)
        // These are mock-tokenizer sidecars, never replacements for model payloads
        // or an assertion that this placeholder is a real tokenizer asset.
        try Data("{}".utf8).write(to: root.appendingPathComponent("tokenizer.json"))
        try JSONSerialization.data(withJSONObject: [
            "chat_template": "embedded alternate", "eos_token": "<|im_end|>",
        ])
        .write(to: root.appendingPathComponent("tokenizer_config.json"))
        try Data(literal.utf8).write(to: root.appendingPathComponent("chat_template.jinja"))
        let native = try JSONDecoder().decode(
            MiMoV26Configuration.self,
            from: Data(contentsOf: root.appendingPathComponent("config.json")))
        try JSONSerialization.data(withJSONObject: [
            "eos_token_id": allEOS ? Array(0 ..< native.vocabularySize) : [4, 5]
        ])
        .write(to: root.appendingPathComponent("generation_config.json"))
        let p = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: Data(contentsOf: fixtures.appendingPathComponent("provenance.json")))
                as? [String: String])
        let provenance = try MiMoV26ConvertedProvenance(
            artifactID: XCTUnwrap(p["artifactID"]),
            sourceRepository: XCTUnwrap(p["sourceRepository"]),
            sourceRevision: XCTUnwrap(p["sourceRevision"]),
            conversionManifestSHA256: XCTUnwrap(p["conversionManifestSHA256"]))
        let plan = try MiMoV26FilesystemWeights.preflight(
            root: root, provenance: provenance,
            limits: .init(maximumShardBytes: 1_048_576, maximumTotalFileBytes: 4_194_304))
        return (root, try MiMoV26SerialLoadSession(plan: plan))
    }
    private func nativeLane() throws {
        guard ProcessInfo.processInfo.environment["MIMO_V26_SERIAL_NATIVE_TESTS"] == "1" else {
            throw XCTSkip("Requires coordinator-owned native lane")
        }
    }
    private func prepared(
        _ root: URL, _ session: MiMoV26SerialLoadSession,
        tokenizer: MockTokenizer = .init(), toolCallFormat: ToolCallFormat? = nil
    ) async throws -> MiMoV26ModelFactory.Prepared {
        var configuration = ResolvedModelConfiguration(directory: root)
        configuration.extraEOSTokens = ["<|extra_eos|>"]
        configuration.toolCallFormat = toolCallFormat
        return try await MiMoV26ModelFactory.prepare(
            request: session.request, configuration: configuration,
            tokenizerLoader: Loader(tokenizer: tokenizer))
    }

    func testPreparationBindsExactRequestTemplateEOSAndNativeDefaults() async throws {
        let (root, session) = try fixture()
        let tokenizer = MockTokenizer()
        let value = try await prepared(root, session, tokenizer: tokenizer)
        XCTAssertEqual(value.request, session.request)
        XCTAssertEqual(value.processor.chatTemplate, literal)
        XCTAssertEqual(value.stopTokenIDs, [3, 4, 5, 7])
        XCTAssertEqual(
            tokenizer.loadedDirectories, [root.resolvingSymlinksInPath().standardizedFileURL])
        XCTAssertEqual(tokenizer.calls.count, 3)
        XCTAssertNil(tokenizer.calls[0].context)
        XCTAssertEqual(tokenizer.calls[1].context?["enable_thinking"] as? Bool, true)
        XCTAssertEqual(tokenizer.calls[2].context?["enable_thinking"] as? Bool, false)
        XCTAssertTrue(tokenizer.calls.allSatisfy { $0.template == literal })
        let other = try MiMoV26SerialLoadSession(
            plan: MiMoV26FilesystemWeights.preflight(
                root: root,
                provenance: try provenance(),
                limits: .init(maximumShardBytes: 1_048_576, maximumTotalFileBytes: 4_194_304)))
        let wrong = Permit(other.request)
        XCTAssertThrowsError(
            try withMiMoConstructionScope { constructionWork in
                try MiMoV26ModelFactory.load(
                    session: other, reservation: wrong, prepared: value, retaining: constructionWork
                )
            })
        XCTAssertEqual(wrong.validations, 0)
    }

    func testTokenizerFailureAndAssetDriftRejectBeforeNativeLoad() async throws {
        let (root, session) = try fixture()
        let tokenizer = MockTokenizer()
        do {
            _ = try await MiMoV26ModelFactory.prepare(
                request: session.request, configuration: .init(directory: root),
                tokenizerLoader: Loader(tokenizer: tokenizer, fail: true))
            XCTFail("tokenizer failure accepted")
        } catch is ProbeError {}
        let value = try await prepared(root, session)
        let permit = Permit(session.request)
        try Data("different".utf8).write(to: root.appendingPathComponent("chat_template.jinja"))
        XCTAssertThrowsError(
            try withMiMoConstructionScope { constructionWork in
                try MiMoV26ModelFactory.load(
                    session: session, reservation: permit, prepared: value,
                    retaining: constructionWork)
            })
        XCTAssertEqual(permit.validations, 0)
        let (secondRoot, second) = try fixture()
        do {
            _ = try await MiMoV26ModelFactory.prepare(
                request: second.request, configuration: .init(directory: secondRoot),
                tokenizerLoader: Loader(
                    tokenizer: .init(),
                    mutateTemplate: secondRoot.appendingPathComponent("chat_template.jinja")))
            XCTFail("tokenizer await changed source without refusal")
        } catch { XCTAssertTrue(error is MiMoV26FactoryError) }
    }

    func testBoundedReaderDetectsGrowthReplacementAndSymlinkWithoutOversizedRead() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mimo-factory-read-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let root = directory.resolvingSymlinksInPath().standardizedFileURL
        let exact = root.appendingPathComponent("exact.json")
        try Data(repeating: 65, count: 16).write(to: exact)
        let exactReader = try MiMoV26FactoryMetadataFile(exact)
        XCTAssertEqual(try exactReader.read(maximumBytes: 16).count, 16)
        XCTAssertEqual(exactReader.bytesRead, 16)
        XCTAssertThrowsError(try exactReader.read(maximumBytes: 16))
        let tooSmall = try MiMoV26FactoryMetadataFile(exact)
        XCTAssertThrowsError(try tooSmall.read(maximumBytes: 15))
        XCTAssertEqual(tooSmall.bytesRead, 0)

        let growing = root.appendingPathComponent("growing.json")
        try Data(repeating: 65, count: 8).write(to: growing)
        let growingReader = try MiMoV26FactoryMetadataFile(growing)
        XCTAssertThrowsError(
            try growingReader.read(
                maximumBytes: 16,
                beforeFirstRead: {
                    // Deterministic mutation after the pre-read validation: no racing
                    // thread or sleep is required to reach the actual read-size bound.
                    let writer = try FileHandle(forWritingTo: growing)
                    defer { try? writer.close() }
                    try writer.seekToEnd()
                    try writer.write(contentsOf: Data(repeating: 66, count: 64))
                }))
        XCTAssertEqual(growingReader.largestReadRequest, 17)
        XCTAssertGreaterThan(growingReader.bytesRead, 0)
        XCTAssertLessThanOrEqual(
            growingReader.bytesRead, 17, "actual read must be limited to maximum+1 after growth")

        let replaced = root.appendingPathComponent("replaced.json")
        try Data("old".utf8).write(to: replaced)
        let replacementReader = try MiMoV26FactoryMetadataFile(replaced)
        XCTAssertThrowsError(
            try replacementReader.read(
                maximumBytes: 16,
                beforeFirstRead: {
                    let next = root.appendingPathComponent("next.json")
                    try Data("new".utf8).write(to: next)
                    try FileManager.default.moveItem(
                        at: replaced, to: root.appendingPathComponent("old.json"))
                    try FileManager.default.moveItem(at: next, to: replaced)
                }))
        XCTAssertLessThanOrEqual(replacementReader.bytesRead, 17)
        let link = root.appendingPathComponent("link.json")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: exact)
        XCTAssertThrowsError(try MiMoV26FactoryMetadataFile(link))
    }

    func testTokenizerSameSizeEditWithRestoredMTimeChangesCTimeAndRefusesLoad() async throws {
        let (root, session) = try fixture()
        let value = try await prepared(root, session)
        let url = root.appendingPathComponent("tokenizer.json").resolvingSymlinksInPath()
            .standardizedFileURL
        let before = try MiMoV26FactoryMetadataFile(url).object
        var raw = stat()
        XCTAssertEqual(lstat(url.path, &raw), 0)
        #if canImport(Darwin)
            let preserved = [raw.st_atimespec, raw.st_mtimespec]
        #else
            let preserved = [raw.st_atim, raw.st_mtim]
        #endif
        let writer = try FileHandle(forWritingTo: url)
        try writer.write(contentsOf: Data("[]".utf8))  // replaces "{}" without changing size or inode
        XCTAssertEqual(
            preserved.withUnsafeBufferPointer { futimens(writer.fileDescriptor, $0.baseAddress) }, 0
        )
        try writer.close()
        let after = try MiMoV26FactoryMetadataFile(url).object
        XCTAssertEqual(after.bytes, before.bytes)
        XCTAssertEqual(after.inode, before.inode)
        XCTAssertEqual(after.modifiedSeconds, before.modifiedSeconds)
        XCTAssertEqual(after.modifiedNanoseconds, before.modifiedNanoseconds)
        XCTAssertTrue(
            after.changedSeconds != before.changedSeconds
                || after.changedNanoseconds != before.changedNanoseconds)
        let permit = Permit(session.request)
        // If the metadata gate regresses, this still blocks native construction.
        permit.revoke()
        XCTAssertThrowsError(
            try withMiMoConstructionScope { constructionWork in
                try MiMoV26ModelFactory.load(
                    session: session, reservation: permit, prepared: value,
                    retaining: constructionWork)
            }
        ) {
            XCTAssertEqual($0 as? MiMoV26FactoryError, .changedTokenizerAssets)
        }
        XCTAssertEqual(permit.validations, 0)
    }

    func testTextInputPreservesToolsAndRejectsEveryMediaRouteBeforeTokenization() throws {
        let tokenizer = MockTokenizer()
        let processor = try MiMoV26TextProcessor(
            tokenizer: tokenizer, chatTemplate: literal,
            vocabularySize: 128, maximumSequenceLength: 128)
        let historicalFunction: [String: String] = ["name": "f", "arguments": "{\"x\": 1}"]
        let historicalCalls: [Message] = [["id": "call-1", "function": historicalFunction]]
        let messages: [Message] = [
            [
                "role": "assistant", "content": "answer", "reasoning_content": "prior thought",
                "tool_calls": historicalCalls,
            ]
        ]
        let tools: [[String: any Sendable]] = [["type": "function", "function": ["name": "f"]]]
        _ = try processor.renderTokens(
            input: UserInput(
                messages: messages, tools: tools,
                additionalContext: ["enable_thinking": false]))
        let call = try XCTUnwrap(tokenizer.calls.last)
        XCTAssertEqual(call.messages[0]["reasoning_content"] as? String, "prior thought")
        XCTAssertEqual(call.tools?.count, 1)
        var chat = Chat.Message.user("per-message tools")
        chat.templateFields["tools"] = tools
        _ = try processor.renderTokens(input: UserInput(chat: [chat]))
        XCTAssertNotNil(tokenizer.calls.last?.messages[0]["tools"])
        var audioChat = Chat.Message.user("audio")
        audioChat.templateFields["audio"] = "unused"
        let callsBefore = tokenizer.calls.count
        let unusedAudio: [String: String] = ["data": "unused", "format": "wav"]
        let audioContent: [Message] = [["type": "input_audio", "input_audio": unusedAudio]]
        for input in [
            UserInput(prompt: "image", images: [.url(URL(fileURLWithPath: "/not-read.png"))]),
            UserInput(prompt: "video", videos: [.frames([])]),
            UserInput(messages: [["role": "user", "content": audioContent]]),
            UserInput(messages: [
                ["role": "user", "content": [["type": "image_url", "image_url": "unused"]]]
            ]),
            UserInput(messages: [
                ["role": "user", "content": [["type": "video", "video": "unused"]]]
            ]),
            UserInput(messages: [["role": "user", "content": "text", "audio": "unused"]]),
            UserInput(chat: [audioChat]),
            UserInput(prompt: "audio", additionalContext: ["audio": "unused"]),
            UserInput(prompt: "bad control", additionalContext: ["enable_thinking": "false"]),
            UserInput(prompt: "bad control", additionalContext: ["enable_thinking": 0]),
        ] {
            XCTAssertThrowsError(try processor.renderTokens(input: input))
        }
        XCTAssertEqual(tokenizer.calls.count, callsBefore)
    }

    func testKnownRoleMissingNullEmptyBodiesPreserveNativeToolHistory() throws {
        let tokenizer = MockTokenizer()
        let processor = try MiMoV26TextProcessor(
            tokenizer: tokenizer, chatTemplate: literal,
            vocabularySize: 128, maximumSequenceLength: 128)
        let bodyVariants: [Message] = [
            [:], ["content": NSNull()],
            ["content": MLXLMCommon.JSONValue.null], ["content": ""],
        ]
        let arguments: [String: any Sendable] = [
            "text": "literal &amp; e\u{301}", "nothing": NSNull(),
        ]
        let calls: [[String: any Sendable]] = [
            [
                "id": "c", "type": "function",
                "function": ["name": "echo", "arguments": arguments] as [String: any Sendable],
            ]
        ]
        let currentTools: [ToolSpec] = [
            ["type": "function", "function": ["name": "different"] as [String: any Sendable]]
        ]
        for role in ["system", "user", "assistant", "tool"] {
            for variant in bodyVariants {
                var message = variant
                message["role"] = role
                if role == "assistant" {
                    message["reasoning_content"] = "retain prior thought"
                    message["tool_calls"] = calls
                }
                for tools: [ToolSpec]? in [nil, currentTools] {
                    var messages = [message]
                    if role == "assistant" {
                        messages.append(["role": "tool", "tool_call_id": "c", "content": "ok"])
                    }
                    _ = try processor.renderTokens(
                        input: UserInput(messages: messages, tools: tools))
                    let delegated = try XCTUnwrap(tokenizer.calls.last)
                    XCTAssertEqual(delegated.messages[0]["role"] as? String, role)
                    XCTAssertEqual(delegated.messages[0]["content"] as? String, "")
                    if role == "assistant" {
                        XCTAssertEqual(
                            delegated.messages[0]["reasoning_content"] as? String,
                            "retain prior thought")
                        let retained = try XCTUnwrap(
                            delegated.messages[0]["tool_calls"] as? [[String: any Sendable]])
                        XCTAssertEqual(retained[0]["id"] as? String, "c")
                        let function = try XCTUnwrap(
                            retained[0]["function"] as? [String: any Sendable])
                        XCTAssertEqual(function["name"] as? String, "echo")
                        let values = try XCTUnwrap(function["arguments"] as? [String: any Sendable])
                        XCTAssertEqual(
                            Data(try XCTUnwrap(values["text"] as? String).utf8),
                            Data("literal &amp; e\u{301}".utf8))
                        XCTAssertTrue(
                            values["nothing"] is NSNull,
                            "Only the content slot may be canonicalized")
                        XCTAssertEqual(delegated.messages[1]["tool_call_id"] as? String, "c")
                    }
                }
            }
        }
        // This spy proves admission/delegation only. Pinned Jinja byte/token
        // parity is prepared in MiMoV26ConsumerParityTests, not mocked here.
    }

    func testEmptyBodyCompatibilityKeepsMalformedUnknownAndMediaRefusals() throws {
        let tokenizer = MockTokenizer()
        let processor = try MiMoV26TextProcessor(
            tokenizer: tokenizer, chatTemplate: literal,
            vocabularySize: 128, maximumSequenceLength: 128)
        let invalid: [Message] = [
            ["role": "unknown"], ["role": 0, "content": NSNull()],
            ["role": "assistant", "content": 17], ["role": "assistant", "content": false],
            ["role": "assistant", "content": ["text": "not a part array"]],
            ["role": "assistant", "content": [1, 2]],
            [
                "role": "assistant",
                "content": [["type": "text", "text": 7] as [String: any Sendable]],
            ],
            ["role": "assistant", "content": [["type": "unknown", "text": "ignored"]]],
            ["role": "assistant", "content": NSNull(), "image": NSNull()],
            [
                "role": "assistant",
                "content": [["type": "input_audio", "input_audio": "not-read"]],
            ],
        ]
        for message in invalid {
            XCTAssertThrowsError(try processor.renderTokens(input: UserInput(messages: [message])))
        }
        XCTAssertTrue(tokenizer.calls.isEmpty)
    }

    func testConflictingEOSAndDifferentTokenizerRootHaveNoLoaderFallback() async throws {
        let (root, session) = try fixture()
        let tokenizer = MockTokenizer()
        var config = ResolvedModelConfiguration(directory: root)
        config.eosTokenIds = [9]
        do {
            _ = try await MiMoV26ModelFactory.prepare(
                request: session.request, configuration: config,
                tokenizerLoader: Loader(tokenizer: tokenizer))
            XCTFail("conflicting EOS accepted")
        } catch { XCTAssertTrue(error is MiMoV26FactoryError) }
        config.eosTokenIds = []
        config.tokenizerDirectory = root.deletingLastPathComponent()
        do {
            _ = try await MiMoV26ModelFactory.prepare(
                request: session.request, configuration: config,
                tokenizerLoader: Loader(tokenizer: tokenizer))
            XCTFail("foreign tokenizer root accepted")
        } catch { XCTAssertTrue(error is MiMoV26FactoryError) }
        XCTAssertTrue(tokenizer.loadedDirectories.isEmpty)
    }

    func testNativeContextHasOneParameterTreeAndRecoverableGenerationGate() async throws {
        try nativeLane()
        let (root, session) = try fixture()
        let value = try await prepared(root, session)
        let context = try withMiMoConstructionScope { constructionWork in
            try MiMoV26ModelFactory.load(
                session: session, reservation: Permit(session.request), prepared: value,
                retaining: constructionWork)
        }
        let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
        XCTAssertEqual(model.loadReceipt.sessionID, session.request.sessionID)
        XCTAssertEqual(model.parameters().flattened().count, model.loadReceipt.parameterCount)
        XCTAssertEqual(model.nativeConfiguration.modelType, "mimo_v2")
        XCTAssertEqual(context.configuration.eosTokenIds, [4, 5])
        XCTAssertEqual(context.configuration.toolCallFormat, .mimoV2)
        XCTAssertEqual(model.stopTokenIDs, [3, 4, 5, 7])
        let input = try await context.processor.prepare(input: UserInput(prompt: "hello"))
        XCTAssertThrowsError(
            try TokenIterator(input: input, model: model, parameters: .init(maxTokens: 2))
        ) {
            XCTAssertEqual($0 as? MiMoV26FactoryError, .nativeCBv2Required)
            XCTAssertTrue($0.localizedDescription.contains("makeCBv2Binding"))
            XCTAssertFalse($0.localizedDescription.contains("sparse"))
            XCTAssertFalse($0.localizedDescription.contains("recurrent"))
        }
        let cache = model.newCache(parameters: nil)
        XCTAssertThrowsError(try model.prepare(input, cache: cache, windowSize: nil))
        XCTAssertThrowsError(
            try withMiMoConstructionScope { constructionWork in
                try MiMoV26ModelFactory.load(
                    session: session, reservation: Permit(session.request), prepared: value,
                    retaining: constructionWork)
            }
        ) {
            XCTAssertEqual($0 as? MiMoV26SerialLoadError, .alreadyConsumed)
        }
    }

    func testNativeBindingRetainsReservationAndAllTowersAfterWrapperRelease() async throws {
        try nativeLane()
        let (root, session) = try fixture()
        let value = try await prepared(root, session)
        var permit: Permit? = Permit(session.request)
        weak var permitWitness = permit
        var context: ModelContext? = try withMiMoConstructionScope { constructionWork in
            try MiMoV26ModelFactory.load(
                session: session,
                reservation: XCTUnwrap(permit), prepared: value, retaining: constructionWork)
        }
        var wrapper = context?.model as? MiMoV26LoadedModel
        weak var visionWitness = wrapper?.resources.loaded.bundle.vision
        weak var audioWitness = wrapper?.resources.loaded.bundle.audioPatch
        var binding: MiMoV26CBv2Binding? = try XCTUnwrap(wrapper).makeCBv2Binding(enableMTP: true)
        var adapter = binding?.adapter
        var assistant = binding?.assistant
        binding = nil
        wrapper = nil
        context = nil
        permit = nil
        XCTAssertNotNil(permitWitness)
        XCTAssertNotNil(visionWitness)
        XCTAssertNotNil(audioWitness)
        XCTAssertNotNil(adapter?.target)
        adapter = nil
        XCTAssertNotNil(assistant?.mtpTargetIdentity)
        XCTAssertNotNil(permitWitness)
        XCTAssertNotNil(visionWitness)
        assistant = nil
        XCTAssertNil(permitWitness)
        XCTAssertNil(visionWitness)
        XCTAssertNil(audioWitness)
    }

    func testNativeCBv2GenerationUsesFactoryEOSAndContainerOwnership() async throws {
        try nativeLane()
        let (root, session) = try fixture(allEOS: true)
        let value = try await prepared(root, session)
        let context = try withMiMoConstructionScope { constructionWork in
            try MiMoV26ModelFactory.load(
                session: session, reservation: Permit(session.request), prepared: value,
                retaining: constructionWork)
        }
        let wrapper = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
        let binding = try wrapper.makeCBv2Binding()
        XCTAssertNil(binding.assistant)
        _ = try withMiMoConstructionScope { constructionWork in
            try binding.adapter.probeNativeKVTypes(retaining: constructionWork)
        }
        let backend = try binding.adapter.makeBackend(bytesCapacity: 16 << 20)
        let engine = EngineV2(
            model: binding.adapter, layerKinds: binding.adapter.layerKinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: binding.adapter.makeCaches()),
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 16, prefillChunkSize: 4,
                maxWaiting: 2))
        let output = await cbv2SchedCollect(
            try engine.submit(
                .init(
                    id: .init(1), promptTokens: [1, 2, 6],
                    sampling: .init(temperature: 0), maxTokens: 4, stopTokens: binding.stopTokenIDs)
            ))
        await engine.shutdown()
        XCTAssertEqual(output.finishReason, .stop)
        XCTAssertEqual(
            output.tokens.count, 1,
            "all-vocabulary synthetic EOS must stop the actual first target token")
        XCTAssertEqual(backend.bytesInUse, 0)
        let (otherRoot, other) = try fixture()
        let otherPrepared = try await prepared(otherRoot, other, toolCallFormat: .json)
        let managedWork = NativeConstructionWork()
        defer {
            if managedWork.snapshot.isRetainedFault { _ = Unmanaged.passRetained(managedWork) }
        }
        let container = try await MiMoV26ModelFactory.loadContainer(
            session: other, reservation: Permit(other.request),
            prepared: otherPrepared, retaining: managedWork)
        try await managedWork.acknowledgeContainerAdoption(container)
        let identity = await container.perform { context in
            (
                (context.model as! MiMoV26LoadedModel).loadReceipt.sessionID,
                context.configuration.toolCallFormat
            )
        }
        XCTAssertEqual(identity.0, other.request.sessionID)
        XCTAssertEqual(
            identity.1, .json,
            "explicit supported parser override must survive factory construction")
    }

    func testNativeFinalCallbackFailureCancellationAndReloadDoNotPublish() async throws {
        try nativeLane()
        for cancel in [false, true] {
            let (root, session) = try fixture()
            let value = try await prepared(root, session)
            var permit: Permit? = Permit(session.request)
            weak var witness = permit
            var reached = false
            var cancelled = false
            XCTAssertThrowsError(
                try withMiMoConstructionScope { constructionWork in
                    try MiMoV26ModelFactory.load(
                        session: session, reservation: XCTUnwrap(permit), prepared: value,
                        retaining: constructionWork,
                        isCancelled: { cancelled },
                        progress: { progress in
                            XCTAssertNotNil(witness)
                            if progress.phase == .complete {
                                reached = true
                                permit = nil
                                XCTAssertNotNil(
                                    witness,
                                    "factory/serial transaction must retain the host owner through unwind"
                                )
                                if cancel { cancelled = true } else { throw ProbeError.callback }
                            }
                        })
                })
            XCTAssertTrue(reached)
            XCTAssertNil(witness, "failed publication must not leak a hidden loaded owner")
            XCTAssertThrowsError(
                try withMiMoConstructionScope { constructionWork in
                    try MiMoV26ModelFactory.load(
                        session: session, reservation: Permit(session.request), prepared: value,
                        retaining: constructionWork)
                }
            ) {
                XCTAssertEqual($0 as? MiMoV26SerialLoadError, .alreadyConsumed)
            }
        }
        let (root, session) = try fixture()
        let value = try await prepared(root, session)
        let permit = Permit(session.request)
        permit.revoke()
        XCTAssertThrowsError(
            try withMiMoConstructionScope { constructionWork in
                try MiMoV26ModelFactory.load(
                    session: session, reservation: permit, prepared: value,
                    retaining: constructionWork)
            })
        XCTAssertEqual(permit.validations, 1)
    }

    func testCustomCancellationGetsAnAdditionalVetoAfterSerialLoadCompletes() async throws {
        try nativeLane()
        let (root, control) = try fixture()
        var complete = false
        var serialPostCompleteCalls = 0
        do {
            // Measure the real serial session's own predicate visits, rather
            // than hard-code its internal revalidation/shard loop count.
            let loaded = try withMiMoConstructionScope { constructionWork in
                try control.load(
                    reservation: Permit(control.request), retaining: constructionWork,
                    isCancelled: {
                        if complete { serialPostCompleteCalls += 1 }
                        return false
                    }, progress: { if $0.phase == .complete { complete = true } })
            }
            StreamOrDevice.default.stream.synchronize()
            withExtendedLifetime(loaded) {}
        }
        let session = try MiMoV26SerialLoadSession(
            plan: MiMoV26FilesystemWeights.preflight(
                root: root,
                provenance: try provenance(),
                limits: .init(maximumShardBytes: 1_048_576, maximumTotalFileBytes: 4_194_304)))
        let value = try await prepared(root, session)
        var completedAgain = false
        var postCompleteCalls = 0
        XCTAssertThrowsError(
            try withMiMoConstructionScope { constructionWork in
                try MiMoV26ModelFactory.load(
                    session: session, reservation: Permit(session.request),
                    prepared: value, retaining: constructionWork,
                    isCancelled: {
                        guard completedAgain else { return false }
                        postCompleteCalls += 1
                        return postCompleteCalls > serialPostCompleteCalls
                    }, progress: { if $0.phase == .complete { completedAgain = true } })
            }
        ) {
            XCTAssertEqual($0 as? MiMoV26SerialLoadError, .cancelled)
        }
        XCTAssertTrue(completedAgain)
        XCTAssertEqual(
            postCompleteCalls, serialPostCompleteCalls + 1,
            "factory must query custom cancellation after its final metadata revalidation")
    }

    private func provenance() throws -> MiMoV26ConvertedProvenance {
        let path = try XCTUnwrap(
            ProcessInfo.processInfo.environment["MIMO_V26_SERIAL_LOAD_FIXTURES"])
        let p = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: Data(
                    contentsOf:
                        URL(fileURLWithPath: path).appendingPathComponent("provenance.json")))
                as? [String: String])
        return try .init(
            artifactID: XCTUnwrap(p["artifactID"]),
            sourceRepository: XCTUnwrap(p["sourceRepository"]),
            sourceRevision: XCTUnwrap(p["sourceRevision"]),
            conversionManifestSHA256: XCTUnwrap(p["conversionManifestSHA256"]))
    }
}
