import Foundation
import MLXHuggingFace
import Testing

@testable import MLXLMCommon

extension UnitTests {

    /// Tests of small value types: the wired memory measurement total and
    /// the Hugging Face downloader error text.
    @Suite
    struct RuntimeValueTests {

        private func measurement(weights: Int, kv: Int, workspace: Int)
            -> WiredMemoryMeasurement
        {
            WiredMemoryMeasurement(
                weightBytes: weights, kvBytes: kv, workspaceBytes: workspace,
                peakActiveBytes: 0, tokenCount: 4, prefillStepSize: 512)
        }

        @Test func totalBytesIsTheSumOfTheParts() {
            #expect(measurement(weights: 100, kv: 20, workspace: 3).totalBytes == 123)
            #expect(measurement(weights: 0, kv: 0, workspace: 0).totalBytes == 0)
        }

        /// A negative part counts as 0.
        @Test func totalBytesIgnoresNegativeParts() {
            #expect(measurement(weights: 100, kv: -20, workspace: 3).totalBytes == 103)
            #expect(measurement(weights: -1, kv: -1, workspace: -1).totalBytes == 0)
        }

        @Test func invalidRepositoryIDDescription() {
            #expect(
                HuggingFaceDownloaderError.invalidRepositoryID("no-slash").errorDescription
                    == "Invalid Hugging Face repository ID: 'no-slash'. Expected format 'namespace/name'."
            )
        }
    }
}
