extension AdmissionV2 {
    /// Optional acceleration must not consume credit promised to later direct
    /// attention calls. Charge the complete candidate as additional nonbackend
    /// memory, atomically, before creating an arena or writing any KV. Its lease
    /// remains independent of request cancellation, capacity changes and other
    /// in-flight steps. Returning it after GPU retirement refunds exactly once.
    func reserveOpportunisticWorkspace(bytes: Int) throws -> CBv2CheckpointReservation {
        try reserveTransient(bytes: bytes)
    }
}
