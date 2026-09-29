import MLX

/// Off-to-the-side restoration. The existing import work owns this value and
/// its temporary promise through the actual fence/retirement callback. It is
/// never installed into a live driver before target adoption also succeeds.
final class CBv2PreparedHistoricalAssistant {
    private let assistant: any CBv2MTPRequestStatefulDrafter
    private var reservation: CBv2CheckpointReservation?
    private(set) var state: (any CBv2MTPRequestState)?
    private(set) var settled = false

    init(assistant: any CBv2MTPRequestStatefulDrafter, reservation: CBv2CheckpointReservation) {
        self.assistant = assistant
        self.reservation = reservation
    }

    func retain(_ state: any CBv2MTPRequestState) { self.state = state }
    func markSettled() { settled = true }

    func takeSettledState() -> (any CBv2MTPRequestState)? {
        guard settled, let state else { return nil }
        self.state = nil
        return state
    }

    /// Only the containing import owner's post-native-completion cleanup may
    /// call this. On failed required completion that callback never runs.
    func closeAfterNativeCompletion() {
        if let state { assistant.releaseRequestState(state) }
        state = nil
        reservation?.release()
        reservation = nil
    }
}

extension CBv2PreparedCompleteCheckpoint {
    /// Run during staged lookup, before enqueue/deadline admission acquires
    /// the native metadata commit lock. Repeated validated lookup is inert.
    func prepareHistoricalAssistant(
        codec: CBv2CompleteCheckpointCodec,
        request: CBv2Request,
        work: CBv2NativeCompletePrefixWork
    ) throws {
        if let historicalAssistantRestoration {
            guard historicalAssistantRestoration.settled else {
                throw CBv2CompleteCheckpointError.incompleteTransfer
            }
            return
        }
        guard let checkpoint else { throw CBv2CompleteCheckpointError.incompleteTransfer }
        guard let encoded = checkpoint.assistant else {
            guard codec.assistant == nil else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            return
        }
        guard codec.contiguousLayout != nil || codec.isNativePagedHistorical,
            let historical = codec.assistant as? any CBv2HistoricalMTPPrefixCheckpointCoding,
            let stateful = codec.assistant as? any CBv2MTPRequestStatefulDrafter,
            let split = codec.assistant as? any CBv2NativeMTPCompletionSplitting,
            checkpoint.position < request.promptTokens.count,
            let descriptors = historical.prefixCheckpointTensorDescriptors(
                targetInputCount: checkpoint.position)
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        let (maximum, overflow) = request.promptTokens.count.addingReportingOverflow(
            max(request.maxTokens, 1))
        guard !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
        // Restored mutable copies coexist with the immutable imported payload.
        // Reserve their bounded native copies plus the full prompt witness
        // before construction. This is not physical-memory/free-space credit.
        let bytes = try CBv2HistoricalMTPCheckpointFootprint.captureBytes(
            position: maximum, descriptors: descriptors)
        let owner = CBv2PreparedHistoricalAssistant(
            assistant: stateful,
            reservation: try codec.admission.reserveTransient(bytes: bytes))
        historicalAssistantRestoration = owner
        try work.retain(owners: [owner])
        try work.captureCurrentStreams()
        do {
            try withError { fault in
                guard let restored = historical.restorePrefixCheckpoint(encoded) else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
                owner.retain(restored)
                try work.retain(owners: [restored])
                try fault.check()
                try stateful.configureRequestState(restored, maximumSequenceLength: maximum)
                try historical.installPrefixCaptureContext(
                    requestState: restored,
                    promptTokens: request.promptTokens)
                let roots = stateful.evaluationTargets(for: restored)
                try work.retain(arrays: roots)
                eval(roots)
                try fault.check()
                try split.fenceRequestStateForNativeCompletion(restored)
                try fault.check()
                try work.fenceForProtectedPromotion()
                try work.commitPreparedAssistant(split, state: restored)
                owner.markSettled()
            }
        } catch {
            // Never refund a possibly submitted restoration on an error path.
            work.requiredCompletionFailed()
            throw error
        }
    }
}
