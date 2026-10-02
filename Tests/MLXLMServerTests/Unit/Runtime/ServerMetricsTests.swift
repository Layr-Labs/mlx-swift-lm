import Foundation
import Testing

@testable import MLXLMServer

extension UnitTests {
    @Suite
    struct ServerMetricsTests {

        private let start = Date(timeIntervalSince1970: 1_000)

        @Test
        func countsRequestsByKindAndUsage() async {
            let metrics = ServerMetrics(startedAt: start)
            await metrics.recordChatRequest()
            await metrics.recordChatRequest()
            await metrics.recordResponseRequest()
            await metrics.recordOtherRequest()
            await metrics.recordError()
            await metrics.recordUsage(.init(promptTokens: 7, completionTokens: 3))
            await metrics.recordUsage(nil)
            await metrics.recordUsage(.init(promptTokens: 1, completionTokens: 2))

            let snapshot = await metrics.snapshot(now: start.addingTimeInterval(12.5))
            #expect(
                snapshot
                    == ServerMetricsSnapshot(
                        requestsTotal: 4, chatCompletionsTotal: 2, responsesTotal: 1,
                        errorsTotal: 1,
                        promptTokensTotal: 8, completionTokensTotal: 5, uptimeSeconds: 12.5))
        }

        @Test
        func prometheusTextListsEveryCounter() async {
            let metrics = ServerMetrics(startedAt: start)
            await metrics.recordResponseRequest()
            let text = await metrics.prometheusText(now: start.addingTimeInterval(2))
            #expect(
                text == """
                    # TYPE mlx_server_requests_total counter
                    mlx_server_requests_total 1
                    # TYPE mlx_server_chat_completions_total counter
                    mlx_server_chat_completions_total 0
                    # TYPE mlx_server_responses_total counter
                    mlx_server_responses_total 1
                    # TYPE mlx_server_errors_total counter
                    mlx_server_errors_total 0
                    # TYPE mlx_server_prompt_tokens_total counter
                    mlx_server_prompt_tokens_total 0
                    # TYPE mlx_server_completion_tokens_total counter
                    mlx_server_completion_tokens_total 0
                    # TYPE mlx_server_uptime_seconds gauge
                    mlx_server_uptime_seconds 2.0

                    """)
        }

        @Test
        func snapshotUsesSnakeCaseKeys() throws {
            let snapshot = ServerMetricsSnapshot(
                requestsTotal: 1, chatCompletionsTotal: 1, responsesTotal: 0, errorsTotal: 0,
                promptTokensTotal: 2, completionTokensTotal: 3, uptimeSeconds: 1)
            let json = String(
                decoding: try JSONEncoder.openAIServer.encode(snapshot), as: UTF8.self)
            #expect(
                json
                    == #"{"chat_completions_total":1,"completion_tokens_total":3,"errors_total":0,"prompt_tokens_total":2,"requests_total":1,"responses_total":0,"uptime_seconds":1}"#
            )
            #expect(
                try JSONDecoder().decode(ServerMetricsSnapshot.self, from: Data(json.utf8))
                    == snapshot)
        }
    }
}
