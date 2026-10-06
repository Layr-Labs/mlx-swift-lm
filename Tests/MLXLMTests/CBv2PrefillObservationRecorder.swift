import Foundation
import MLXLMCommon

/// Numeric callback capture shared by real-engine prompt-completion regressions.
final class CBv2PrefillObservationRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [CBv2Usage] = []
    func append(_ usage: CBv2Usage) { lock.withLock { values.append(usage) } }
    var snapshot: [CBv2Usage] { lock.withLock { values } }
}
