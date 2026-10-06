import Testing

@testable import Qwen38DFlash2

@Suite("Qwen 3.8 runner wired-memory construction")
struct Qwen38WiredMemoryContractTests {
    @Test("derives the post-warm limit and preserves the device safety margin")
    func derivedLimit() {
        #expect(
            Qwen38WiredMemoryContract.limit(
                activeBytes: 1_000,
                recommendedMaximum: 1_000_000_000,
                physicalMemoryBytes: 128 * 1_024 * 1_024 * 1_024)
                == 1_000 + 64 * 1_024 * 1_024)
        #expect(
            Qwen38WiredMemoryContract.limit(
                activeBytes: 99_000_000,
                recommendedMaximum: 100_000_000,
                physicalMemoryBytes: 128 * 1_024 * 1_024 * 1_024)
                == nil)
    }

    @Test("row 50 stays disabled below the measured 96 GiB machine class")
    func minimumMachineClass() {
        #expect(
            Qwen38WiredMemoryContract.limit(
                activeBytes: 1_000,
                recommendedMaximum: nil,
                physicalMemoryBytes: 64 * 1_024 * 1_024 * 1_024)
                == nil)
    }
}
