import MLX

/// The native import loan retains this owner before any assistant copy graph.
/// Its actual stage-funded context becomes request-funded in the same commit
/// that installs pages and KDA state. No state is visible before that commit.
final class CBv2GLM5NextPreparedAssistant {
    private let assistant: any CBv2MTPPrefixCheckpointCoding
    private let proof: CBv2MTPCheckpointImportProof
    private(set) var state: (any CBv2MTPRequestState)?
    private(set) var recurrent: CBv2RecurrentRequestState?
    private(set) var settled = false
    private var committed = false

    init(assistant: any CBv2MTPPrefixCheckpointCoding, proof: CBv2MTPCheckpointImportProof) {
        self.assistant = assistant
        self.proof = proof
    }

    func prepare(checkpoint: CBv2RecurrentCheckpoint, spec: CBv2RecurrentStateSpec,
                 issuer: CBv2MTPAdmissionIssuer, work: CBv2NativeCompletePrefixWork) throws {
        guard state == nil, !settled, let encoded = checkpoint.assistant,
            let admitted = assistant as? any CBv2AdmittedMTPRequestStatefulDrafter,
            let split = assistant as? any CBv2NativeMTPCompletionSplitting else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        recurrent = try CBv2RecurrentRequestState(spec: spec, adoptedCommitted: checkpoint.layers)
        guard let restored = assistant.restorePrefixCheckpoint(encoded) else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        state = restored
        try work.retain(owners: [restored])
        let context = try issuer.context(id: proof.requestID, maximumInputs: proof.maximumInputs, importProof: proof)
        try work.captureCurrentStreams()
        do {
            try withError { fault in
                try admitted.configureRequestState(restored, admission: context)
                let roots = assistant.evaluationTargets(for: restored)
                try work.retain(arrays: roots)
                eval(roots)
                try fault.check()
                try split.fenceRequestStateForNativeCompletion(restored)
                try fault.check()
                try work.fenceForProtectedPromotion()
            }
            settled = true
        } catch {
            work.requiredCompletionFailed()
            throw error
        }
    }

    /// Metadata-only, under the enclosing healthy target publication commit.
    func commit(work: CBv2NativeCompletePrefixWork) throws {
        guard settled, !committed, let state,
            let split = assistant as? any CBv2NativeMTPCompletionSplitting else {
            throw CBv2CompleteCheckpointError.incompleteTransfer
        }
        do {
            try work.commitPreparedAssistant(split, state: state)
            guard state.committedInputCount == proof.ownerPosition - 1,
                (state as? any CBv2MTPRequestResidencyReporting)?.hasUnmeasuredResidency == false else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
        } catch {
            work.requiredCompletionFailed()
            throw error
        }
        committed = true
    }

    func take() -> (CBv2RecurrentRequestState, any CBv2MTPRequestState)? {
        guard committed, let recurrent, let state else { return nil }
        self.recurrent = nil
        self.state = nil
        return (recurrent, state)
    }

    func closeAfterNativeCompletion() throws {
        if let state { assistant.releaseRequestState(state) }
        state = nil
        try recurrent?.release()
        recurrent = nil
        try proof.stage.admission.abandonImportedMTPBinding(proof)
    }
}
