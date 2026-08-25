import Foundation
import Testing

@testable import Qwen38DFlash2

@Suite("Qwen 3.8 DFlash2 runner options")
struct Qwen38DFlash2RunnerOptionsTests {
    @Test("parses the pinned local runner surface")
    func parse() throws {
        let options = try Qwen38RunnerOptions.parse([
            "--target", "/models/target",
            "--draft", "/models/draft",
            "--mode", "dflash2",
            "--prompt", "hello",
            "--max-tokens", "128",
            "--conditioner-tokens", "1024",
            "--receipt", "/tmp/receipt.json",
        ])
        #expect(options.targetPath == "/models/target")
        #expect(options.draftPath == "/models/draft")
        #expect(options.mode == .dflash2)
        #expect(options.prompt == "hello")
        #expect(options.maxTokens == 128)
        #expect(options.conditionerTokens == 1024)
        #expect(options.receiptPath == "/tmp/receipt.json")
        #expect(options.dflashPhysicalWidth == nil)
    }

    @Test("same-prompt conditioner is explicit and positive")
    func conditionerTokens() throws {
        let ordinary = try Qwen38RunnerOptions.parse([
            "--target", "/models/target",
            "--draft", "/models/draft",
            "--prompt", "hello",
        ])
        #expect(ordinary.conditionerTokens == 0)

        #expect(throws: Qwen38RunnerOptionError.self) {
            try Qwen38RunnerOptions.parse([
                "--target", "/models/target",
                "--draft", "/models/draft",
                "--prompt", "hello",
                "--conditioner-tokens", "0",
            ])
        }
    }

    @Test("cycle diagnostics are an explicit non-measured DFlash route")
    func diagnosticCycles() throws {
        let options = try Qwen38RunnerOptions.parse([
            "--target", "/models/target",
            "--draft", "/models/draft",
            "--prompt", "hello",
            "--receipt", "/tmp/diagnostic.json",
            "--diagnostic-cycles",
        ])
        #expect(options.diagnosticCycles)

        #expect(throws: Qwen38RunnerOptionError.self) {
            try Qwen38RunnerOptions.parse([
                "--target", "/models/target",
                "--mode", "ar",
                "--prompt", "hello",
                "--receipt", "/tmp/diagnostic.json",
                "--diagnostic-cycles",
            ])
        }
        #expect(throws: Qwen38RunnerOptionError.self) {
            try Qwen38RunnerOptions.parse([
                "--target", "/models/target",
                "--draft", "/models/draft",
                "--prompt", "hello",
                "--diagnostic-cycles",
            ])
        }
    }

    @Test("prefetched cycle diagnostics exercise the production scheduling route")
    func diagnosticPrefetch() throws {
        let options = try Qwen38RunnerOptions.parse([
            "--target", "/models/target",
            "--draft", "/models/draft",
            "--prompt", "hello",
            "--receipt", "/tmp/prefetched-diagnostic.json",
            "--diagnostic-prefetch",
        ])
        #expect(options.diagnosticPrefetch)
        #expect(!options.diagnosticCycles)

        #expect(throws: Qwen38RunnerOptionError.self) {
            try Qwen38RunnerOptions.parse([
                "--target", "/models/target",
                "--mode", "ar",
                "--prompt", "hello",
                "--receipt", "/tmp/prefetched-diagnostic.json",
                "--diagnostic-prefetch",
            ])
        }
        #expect(throws: Qwen38RunnerOptionError.self) {
            try Qwen38RunnerOptions.parse([
                "--target", "/models/target",
                "--draft", "/models/draft",
                "--prompt", "hello",
                "--receipt", "/tmp/prefetched-diagnostic.json",
                "--diagnostic-cycles",
                "--diagnostic-prefetch",
            ])
        }
    }

    @Test("fixed DFlash width is a construction-time benchmark control")
    func fixedDFlashWidth() throws {
        let options = try Qwen38RunnerOptions.parse([
            "--target", "/models/target",
            "--draft", "/models/draft",
            "--mode", "benchmark",
            "--prompt", "hello",
            "--dflash-width", "6",
        ])
        #expect(options.dflashPhysicalWidth == 6)
        #expect(throws: Qwen38RunnerOptionError.self) {
            try Qwen38RunnerOptions.parse([
                "--target", "/models/target",
                "--draft", "/models/draft",
                "--prompt", "hello",
                "--dflash-width", "9",
            ])
        }
    }

    @Test("rejects an ambiguous prompt source before model construction")
    func promptExclusivity() {
        #expect(throws: Qwen38RunnerOptionError.self) {
            try Qwen38RunnerOptions.parse([
                "--target", "/models/target",
                "--draft", "/models/draft",
                "--prompt", "hello",
                "--prompt-file", "/tmp/prompt.txt",
            ])
        }
    }

    @Test("AR control does not require a draft artifact")
    func autoregressiveWithoutDraft() throws {
        let options = try Qwen38RunnerOptions.parse([
            "--target", "/models/target",
            "--mode", "ar",
            "--prompt", "hello",
        ])
        #expect(options.draftPath == nil)
    }

    @Test("DFlash2 requires its pinned draft artifact")
    func dflashRequiresDraft() {
        #expect(throws: Qwen38RunnerOptionError.self) {
            try Qwen38RunnerOptions.parse([
                "--target", "/models/target",
                "--mode", "dflash2",
                "--prompt", "hello",
            ])
        }
    }

    @Test("validates prompt token IDs before model execution")
    func promptTokenRange() throws {
        try validateQwen38PromptTokenIDs([0, 248_319])
        #expect(throws: Qwen38RunnerOptionError.self) {
            try validateQwen38PromptTokenIDs([-1])
        }
        #expect(throws: Qwen38RunnerOptionError.self) {
            try validateQwen38PromptTokenIDs([248_320])
        }
    }

    @Test("token files reject a malformed field instead of silently dropping it")
    func strictTokenFileParsing() throws {
        #expect(try decodeQwen38TokenIDs(Data("1, 2\n3".utf8)) == [1, 2, 3])
        #expect(try decodeQwen38TokenIDs(Data("[4,5]".utf8)) == [4, 5])
        #expect(throws: Qwen38RunnerOptionError.self) {
            try decodeQwen38TokenIDs(Data("1, nope, 3".utf8))
        }
    }
}
