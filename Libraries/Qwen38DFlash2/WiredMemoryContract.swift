public enum Qwen38WiredMemoryContract {
    public static let headroomBytes = 64 * 1_024 * 1_024
    public static let recommendedReserveBytes = 256 * 1_024 * 1_024
    public static let minimumPhysicalMemoryBytes = 96 * 1_024 * 1_024 * 1_024

    public static func limit(
        activeBytes: Int,
        recommendedMaximum: Int?,
        physicalMemoryBytes: UInt64
    ) -> Int? {
        guard physicalMemoryBytes >= UInt64(minimumPhysicalMemoryBytes), activeBytes > 0 else {
            return nil
        }
        let active = activeBytes
        let (sum, overflow) = active.addingReportingOverflow(headroomBytes)
        let desired = overflow ? Int.max : sum
        guard let recommendedMaximum, recommendedMaximum > 0 else { return desired }
        let ceiling = max(0, recommendedMaximum - recommendedReserveBytes)
        let result = min(desired, ceiling)
        return result > 0 ? result : nil
    }
}
