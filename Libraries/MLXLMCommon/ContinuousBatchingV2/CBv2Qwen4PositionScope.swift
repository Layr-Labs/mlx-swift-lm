/// Request-owned position provenance for a synchronous Qwen4 graph build.
///
/// A mixed decode rectangle fills absent text positions with a ramp beside
/// explicit media positions. Qwen4 must still use its original text rotary
/// path for those absent rows. Equal media planes are not a text-only proof.
/// The scope carries only immutable host booleans, never tensors or request IDs;
/// it restores the caller's binding when the synchronous forward returns or
/// throws. GPU array and cache lifetimes remain owned by the existing engine.
public enum CBv2Qwen4PositionScope {
    @TaskLocal private static var explicitRows: [Bool]?

    public static func withExplicitRows<Result>(
        _ rows: [Bool], operation: () throws -> Result
    ) rethrows -> Result {
        precondition(!rows.isEmpty)
        return try $explicitRows.withValue(rows, operation: operation)
    }

    public static func explicitPosition(row: Int, batch: Int) -> Bool? {
        guard let explicitRows else { return nil }
        precondition(explicitRows.count == batch && row >= 0 && row < batch,
                     "Qwen4 position provenance must match the current row order")
        return explicitRows[row]
    }
}
