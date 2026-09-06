import Foundation
import MLX

/// One throwaway pass through the loaded runner before the resident binds
/// its socket.
///
/// The first forwards of a fresh process are slow for reasons that are not
/// the model: Metal pipelines are built on first use, the allocator grows
/// to its working size, the n-gram map faults its first pages. On the 125B
/// consecutive prefills of one prompt read 3.56 → 0.87 → 0.635 ms per token
/// and then hold within 0.2%. benchd times the first prompt of a window, so
/// a resident that has not run anything hands that ramp to the timed leg.
///
/// The pass is FIXED: one prefill of `prefillTokens` tokens and `decodeSteps`
/// single-token forwards, then the session is released and the allocator
/// drained, so the first phase attaches to a resident in the same state a
/// later phase would. It is not a loop that waits for a number to settle.
public enum BenchWorkerResidentWarm {
    public static let prefillTokens = 1024
    public static let decodeSteps = 8

    /// The token ids of the warm prefill: a fixed cycle over small ids,
    /// which every vocabulary has. The values do not matter; the shapes do.
    static func tokens(count: Int) -> [Int] {
        (0 ..< count).map { 1 + ($0 * 7) % 997 }
    }

    /// Run the pass. Returns the wall seconds it took.
    @discardableResult
    public static func run(
        runner: any Runner,
        memory: any WorkerMemoryReporter = MLXMemoryReporter(),
        prefillTokens: Int = prefillTokens,
        decodeSteps: Int = decodeSteps
    ) throws -> Double {
        let started = DispatchTime.now().uptimeNanoseconds
        try autoreleasepool {
            let stepper = try runner.makeStepper()
            try stepper.begin()
            var next = try stepper.forward(tokens(count: prefillTokens)).argmax
            for _ in 0 ..< decodeSteps {
                next = try stepper.forward([next]).argmax
            }
        }
        memory.drain()
        return Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
    }
}
