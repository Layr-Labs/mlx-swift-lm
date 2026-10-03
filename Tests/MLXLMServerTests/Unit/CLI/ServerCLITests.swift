import Foundation
import Testing

@testable import MLXLMServer

extension UnitTests {
    @Suite
    struct ServerCLITests {

        private func parse(
            _ arguments: [String], environment: [String: String] = [:]
        ) throws -> MLXServerCLICommand {
            try MLXServerCLI.parse(arguments: ["mlx-server"] + arguments, environment: environment)
        }

        @Test
        func defaultsApplyWithoutArgumentsOrEnvironment() throws {
            #expect(
                try parse([])
                    == .run(
                        MLXServerConfiguration(
                            model: "mlx-community/Qwen3-0.6B-4bit", revision: "main",
                            host: "127.0.0.1", port: 8080)))
        }

        @Test
        func optionsSetEachField() throws {
            let command = try parse([
                "--model", "org/m", "--revision", "v1", "--host", "0.0.0.0", "--port", "9000",
                "--model-type", "qwen3", "--tool-call-parser", "json",
                "--reasoning-parser", "deepseek-r1", "--embedding-model", "org/e",
            ])
            #expect(
                command
                    == .run(
                        MLXServerConfiguration(
                            model: "org/m", revision: "v1", host: "0.0.0.0", port: 9000,
                            modelType: "qwen3", toolCallParser: "json",
                            reasoningParser: .deepseekR1, embeddingModel: "org/e")))
        }

        @Test
        func environmentGivesDefaultsAndOptionsWin() throws {
            let environment = [
                "MLX_SERVER_MODEL": "env/model",
                "MLX_SERVER_REVISION": "env-rev",
                "MLX_SERVER_HOST": "10.0.0.1",
                "MLX_SERVER_PORT": "7000",
                "MLX_SERVER_MODEL_TYPE": "gemma4",
                "MLX_SERVER_TOOL_CALL_PARSER": "gemma",
                "MLX_SERVER_REASONING_PARSER": "gpt-oss",
                "MLX_SERVER_EMBEDDING_MODEL": "env/embed",
            ]
            #expect(
                try parse([], environment: environment)
                    == .run(
                        MLXServerConfiguration(
                            model: "env/model", revision: "env-rev", host: "10.0.0.1",
                            port: 7000, modelType: "gemma4", toolCallParser: "gemma",
                            reasoningParser: .harmony, embeddingModel: "env/embed")))

            guard
                case .run(let configuration) = try parse(
                    ["-m", "cli/model"], environment: environment)
            else {
                Issue.record("Expected a run command")
                return
            }
            #expect(configuration.model == "cli/model")
            #expect(configuration.port == 7000)
        }

        @Test
        func aPortInTheEnvironmentThatIsNotANumberGivesTheDefault() throws {
            guard
                case .run(let configuration) = try parse([], environment: ["MLX_SERVER_PORT": "x"])
            else {
                Issue.record("Expected a run command")
                return
            }
            #expect(configuration.port == 8080)
        }

        @Test
        func aPositionalArgumentIsTheModel() throws {
            guard case .run(let configuration) = try parse(["org/positional"]) else {
                Issue.record("Expected a run command")
                return
            }
            #expect(configuration.model == "org/positional")
        }

        @Test
        func helpAndListRoutesStopTheParse() throws {
            #expect(try parse(["--help", "--port", "bad"]) == .help)
            #expect(try parse(["-h"]) == .help)
            #expect(try parse(["--list-routes"]) == .listRoutes)
            #expect(MLXServerCLI.help.hasPrefix("Usage: mlx-server"))
        }

        @Test
        func invalidArgumentsThrow() {
            #expect(throws: MLXServerCLIError.missingValue("--model")) {
                try parse(["--model"])
            }
            #expect(throws: MLXServerCLIError.invalidPort("80a")) {
                try parse(["--port", "80a"])
            }
            #expect(throws: MLXServerCLIError.unknownOption("--verbose")) {
                try parse(["--verbose"])
            }
            #expect(throws: DecodingError.self) {
                try parse(["--reasoning-parser", "unknown"])
            }
            #expect(throws: DecodingError.self) {
                try parse([], environment: ["MLX_SERVER_REASONING_PARSER": "unknown"])
            }
        }

        @Test
        func errorDescriptionsNameTheValue() {
            #expect(
                MLXServerCLIError.missingValue("--host").errorDescription
                    == "Missing value for --host")
            #expect(MLXServerCLIError.invalidPort("p").errorDescription == "Invalid port 'p'")
            #expect(MLXServerCLIError.unknownOption("-x").errorDescription == "Unknown option '-x'")
            #expect(
                MLXServerCLIError.invalidEngineKind("k").errorDescription
                    == "Invalid engine kind 'k'")
        }

        @Test
        func modelConfigurationUsesAnIDWhenNoFolderExists() {
            let configuration = MLXServer.modelConfiguration(
                for: "org/not-a-folder-\(UUID().uuidString)", revision: "r2")
            guard case .id(let id, let revision) = configuration.id else {
                Issue.record("Expected an id configuration")
                return
            }
            #expect(id.hasPrefix("org/not-a-folder-"))
            #expect(revision == "r2")
        }

        @Test
        func modelConfigurationExpandsTheHomeFolder() {
            let home = FileManager.default.homeDirectoryForCurrentUser
            #expect(
                MLXServer.modelConfiguration(for: "~").id
                    == .directory(URL(fileURLWithPath: home.path)))

            // A path under the home folder that does not exist stays an id,
            // with the original text.
            let missing = "~/mlx-server-unit-\(UUID().uuidString)"
            #expect(MLXServer.modelConfiguration(for: missing).id == .id(missing, revision: "main"))
        }

        @Test
        func modelConfigurationUsesAnExistingFolder() throws {
            let directory = FileManager.default.temporaryDirectory
                .appending(path: "mlx-server-unit-\(UUID().uuidString)")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let file = directory.appending(component: "file.txt")
            try Data("x".utf8).write(to: file)

            #expect(
                MLXServer.modelConfiguration(for: directory.path).id
                    == .directory(URL(fileURLWithPath: directory.path)))
            // A file is not a model folder.
            #expect(
                MLXServer.modelConfiguration(for: file.path).id == .id(file.path, revision: "main"))
        }
    }
}
