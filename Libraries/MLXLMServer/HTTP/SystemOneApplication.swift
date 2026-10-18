// Copyright © 2026 Eigen Labs Inc.
import Foundation
import Hummingbird
import MLXDecisions

extension MLXServerApplication {
    /// Dedicated local decision server. The inference gate admits one active request and
    /// rejects excess traffic before collecting bodies, avoiding an unbounded actor queue.
    public static func buildSystemOneApplication(
        modelID: String = "laya",
        host: String = "127.0.0.1",
        port: Int = 8080,
        predict: @Sendable @escaping (Data) async throws -> Data
    ) -> Application<RouterResponder<BasicRequestContext>> {
        Application(
            router: buildSystemOneRouter(modelID: modelID, predict: predict),
            configuration: .init(
                address: .hostname(host, port: port),
                serverName: MLXLMServer.defaultServerName))
    }

    public static func buildSystemOneRouter(
        modelID: String = "laya",
        predict: @Sendable @escaping (Data) async throws -> Data
    ) -> Router<BasicRequestContext> {
        let router = Router()
        let gate = SystemOneRequestGate()
        for path in ["/health", "/v1/health"] {
            router.get(RouterPath(path)) { _, _ async throws -> Response in
                try systemOneJSON(ServerHealthResponse())
            }
        }
        for path in ["/models", "/v1/models"] {
            router.get(RouterPath(path)) { _, _ async throws -> Response in
                let value: DecisionJSON = .object([
                    .init("object", .string("list")),
                    .init(
                        "data",
                        .array([
                            .object([
                                .init("id", .string(modelID)), .init("object", .string("model")),
                                .init("owned_by", .string("local")),
                            ])
                        ])),
                ])
                return systemOneData(Data(value.rendered().utf8))
            }
        }
        let handler: @Sendable (Request, BasicRequestContext) async throws -> Response = {
            request, _ in
            guard gate.acquire() else {
                return try systemOneError(
                    "Laya is processing another request", status: .tooManyRequests,
                    type: "rate_limit_error", code: "model_busy")
            }
            defer { gate.release() }
            do {
                let buffer = try await request.body.collect(upTo: 1 << 20)
                let data = Data(buffer.readableBytesView)
                let parsed = try SystemOneRequest(data: data)
                guard parsed.model == modelID else {
                    return try systemOneError(
                        "Requested model is not loaded", status: .notFound,
                        type: "invalid_request_error", code: "model_not_found")
                }
                return systemOneData(try await predict(data))
            } catch LayaError.invalidJSON(let message) {
                return try systemOneError(
                    message, status: .badRequest,
                    type: "invalid_request_error", code: "invalid_request")
            } catch LayaError.invalidRequest(let message) {
                return try systemOneError(
                    message, status: .unprocessableContent,
                    type: "invalid_request_error", code: "invalid_request")
            } catch let error as any HTTPResponseError {
                throw error
            } catch {
                return try systemOneError(
                    "Laya inference could not complete", status: .internalServerError,
                    type: "server_error", code: "inference_failed")
            }
        }
        router.post("/v1/systemone", use: handler)
        router.post("/systemone", use: handler)
        return router
    }
}

private final class SystemOneRequestGate: @unchecked Sendable {
    private let lock = NSLock()
    private var active = false
    func acquire() -> Bool {
        lock.withLock {
            guard !active else { return false }
            active = true
            return true
        }
    }
    func release() { lock.withLock { active = false } }
}

private func systemOneData(_ data: Data, status: HTTPResponse.Status = .ok) -> Response {
    Response(
        status: status, headers: [.contentType: "application/json"],
        body: .init(byteBuffer: ByteBuffer(bytes: data)))
}

private func systemOneJSON<T: Encodable>(_ value: T, status: HTTPResponse.Status = .ok) throws
    -> Response
{
    systemOneData(try JSONEncoder().encode(value), status: status)
}

private func systemOneError(
    _ message: String, status: HTTPResponse.Status, type: String,
    code: String
) throws -> Response {
    var response = try systemOneJSON(
        OpenAIErrorResponse(message: message, type: type, code: code), status: status)
    if status == .tooManyRequests { response.headers[.retryAfter] = "1" }
    return response
}
