/// The generic backend seam permits two arbitrary submitted waves. EngineV2
/// opts into its stricter scheduling contract: a successor is pure decode,
/// and MTP rounds are finalized before constructing any following step.
enum CBv2WorkspaceOverlapPolicy: Sendable {
    case twoArbitrarySteps
    case engineSerialPrefill

    func bytes(prefill: Int, decode: Int, serialDecodeCalls: Int, sharedArena: Int = 0) -> Int? {
        guard prefill >= 0, decode >= 0, serialDecodeCalls > 0, sharedArena >= 0 else { return nil }
        let (verification, verifyOverflow) = decode.multipliedReportingOverflow(by: serialDecodeCalls)
        guard !verifyOverflow else { return nil }
        switch self {
        case .twoArbitrarySteps:
            guard let wave = Self.sum([max(prefill, verification), sharedArena]) else { return nil }
            let (result, overflow) = wave.multipliedReportingOverflow(by: 2)
            return overflow ? nil : result
        case .engineSerialPrefill:
            guard let prefillAndDecode = Self.sum([prefill, decode, sharedArena, sharedArena]),
                let twoDecode = Self.sum([decode, decode, sharedArena, sharedArena]),
                let verifyAndArena = Self.sum([verification, sharedArena])
            else { return nil }
            return max(prefillAndDecode, max(twoDecode, verifyAndArena))
        }
    }

    func envelope(
        prefill: CBv2WorkspaceCostEnvelope, decode: CBv2WorkspaceCostEnvelope,
        serialDecodeCalls: Int, sharedArena: CBv2WorkspaceCostEnvelope = .init()
    ) -> CBv2WorkspaceCostEnvelope? {
        guard serialDecodeCalls > 0, let verification = decode.scaled(by: serialDecodeCalls)
        else { return nil }
        switch self {
        case .twoArbitrarySteps:
            return prefill.covering(verification).adding(sharedArena)?.scaled(by: 2)
        case .engineSerialPrefill:
            guard let twoArenas = sharedArena.scaled(by: 2),
                let prefillAndDecode = prefill.adding(decode)?.adding(twoArenas),
                let twoDecode = decode.scaled(by: 2)?.adding(twoArenas),
                let verifyAndArena = verification.adding(sharedArena)
            else { return nil }
            return prefillAndDecode.covering(twoDecode).covering(verifyAndArena)
        }
    }

    private static func sum(_ values: [Int]) -> Int? {
        values.reduce(Optional(0)) { result, value in
            guard let result else { return nil }
            let (sum, overflow) = result.addingReportingOverflow(value)
            return overflow ? nil : sum
        }
    }
}
