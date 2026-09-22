// Copyright © 2026 Eigen Labs Inc.
import Foundation
import Hummingbird
import HummingbirdTesting
import Testing

@testable import MLXLMServer

struct SystemOneHTTPTests {
    private static let request =
        #"{"model":"laya","state":"x","questions":{"q":{"type":"noul","instructions":"Valid?"}}}"#

    @Test func returnsDecisionResponseAndRejectsBadRequestsBeforeInference() async throws {
        let calls = SystemOneCallRecorder()
        let app = MLXServerApplication.buildSystemOneApplication(port: 0) { data in
            await calls.record()
            return Data(
                #"{"model":"laya","answers":{"q":{"type":"noul","noul":0.9}},"usage":{"input_tokens":14,"output_tokens":0}}"#
                    .utf8)
        }
        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/systemone", method: .post,
                body: ByteBuffer(string: Self.request)
            ) { response in
                #expect(response.status == .ok)
                #expect(String(buffer: response.body).contains(#""output_tokens":0"#))
            }
            try await client.execute(
                uri: "/v1/systemone", method: .post,
                body: ByteBuffer(string: "{")
            ) { response in
                #expect(response.status == .badRequest)
            }
            for text in [
                Self.request.replacingOccurrences(of: #""state":"x""#, with: #""state":true"#),
                Self.request.replacingOccurrences(
                    of: #""state":"x""#, with: #""stream":true,"state":"x""#),
            ] {
                try await client.execute(
                    uri: "/v1/systemone", method: .post,
                    body: ByteBuffer(string: text)
                ) { response in
                    #expect(response.status == .unprocessableContent)
                }
            }
            try await client.execute(
                uri: "/v1/systemone", method: .post,
                body: ByteBuffer(
                    string: Self.request.replacingOccurrences(of: "laya", with: "other"))
            ) { response in
                #expect(response.status == .notFound)
            }
            try await client.execute(
                uri: "/v1/systemone", method: .post,
                body: ByteBuffer(string: String(repeating: "x", count: (1 << 20) + 1))
            ) { response in
                #expect(response.status == .contentTooLarge)
            }
        }
        #expect(await calls.count == 1)
    }

    @Test func rejectsExcessConcurrentRequestAndRecoversAfterCompletion() async throws {
        let barrier = SystemOneInferenceBarrier()
        let app = MLXServerApplication.buildSystemOneApplication(port: 0) { _ in
            await barrier.run()
            return Data("{}".utf8)
        }
        try await app.test(.router) { client in
            let first = Task {
                try await client.execute(
                    uri: "/v1/systemone", method: .post,
                    body: ByteBuffer(string: Self.request)
                ) { response in
                    #expect(response.status == .ok)
                }
            }
            await barrier.waitForStart()
            try await client.execute(
                uri: "/v1/systemone", method: .post,
                body: ByteBuffer(string: Self.request)
            ) { response in
                #expect(response.status == .tooManyRequests)
                #expect(response.headers[.retryAfter] == "1")
            }
            await barrier.finish()
            try await first.value
            try await client.execute(
                uri: "/v1/systemone", method: .post,
                body: ByteBuffer(string: Self.request)
            ) { response in
                #expect(response.status == .ok)
            }
        }
    }

    @Test func failureDoesNotLeakDetailsOrKeepAdmissionOccupied() async throws {
        let calls = SystemOneCallRecorder()
        let app = MLXServerApplication.buildSystemOneApplication(port: 0) { _ in
            await calls.record()
            if await calls.count == 1 { throw NSError(domain: "private-checkpoint-path", code: 1) }
            return Data("{}".utf8)
        }
        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/systemone", method: .post,
                body: ByteBuffer(string: Self.request)
            ) { response in
                #expect(response.status == .internalServerError)
                #expect(!String(buffer: response.body).contains("private-checkpoint-path"))
            }
            try await client.execute(
                uri: "/v1/systemone", method: .post,
                body: ByteBuffer(string: Self.request)
            ) { response in
                #expect(response.status == .ok)
            }
        }
    }
}

private actor SystemOneCallRecorder {
    var count = 0
    func record() { count += 1 }
}

private actor SystemOneInferenceBarrier {
    private var started = false
    private var complete = false
    private var startWaiter: CheckedContinuation<Void, Never>?
    private var finishWaiter: CheckedContinuation<Void, Never>?
    func run() async {
        started = true
        startWaiter?.resume()
        startWaiter = nil
        if !complete { await withCheckedContinuation { finishWaiter = $0 } }
    }
    func waitForStart() async {
        if !started { await withCheckedContinuation { startWaiter = $0 } }
    }
    func finish() {
        complete = true
        finishWaiter?.resume()
        finishWaiter = nil
    }
}
