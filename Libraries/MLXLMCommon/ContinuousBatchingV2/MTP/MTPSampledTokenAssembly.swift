// Copyright © 2026 Eigen Labs.

/// Reassemble already-sampled rows in scheduler order. This helper does not
/// sample, evaluate tensors, or touch acceptance/history/retirement state.
/// The homogeneous-decode fast path comes from the final Gemma MLXFast port;
/// generic tokens keep the ordering contract testable without a GPU runtime.
enum CBv2MTPSampledTokenAssembly {
    static func assemble<Work, ID: Hashable, Tokens>(
        work: [Work], decodeCount: Int, decodeTokens: Tokens?,
        prefillTokens: [ID: Tokens],
        id: (Work) -> ID, isDecode: (Work) -> Bool,
        slice: (Tokens, Range<Int>) -> Tokens,
        concatenate: ([Tokens]) -> Tokens
    ) -> (rows: [ID], tokens: Tokens?) {
        // decodeCount is the count of the stable isDecode filter used by the
        // caller for sampling. No non-decode work can be omitted by this test.
        if decodeCount == work.count, let decodeTokens {
            return (work.map(id), decodeTokens)
        }
        var pieces: [Tokens] = []
        var rows: [ID] = []
        var decodeIndex = 0
        for row in work {
            if isDecode(row) {
                precondition(decodeTokens != nil, "MTP decode rows require sampled tokens")
                pieces.append(slice(decodeTokens!, decodeIndex ..< decodeIndex + 1))
                decodeIndex += 1
                rows.append(id(row))
            } else if let sampled = prefillTokens[id(row)] {
                pieces.append(sampled)
                rows.append(id(row))
            }
        }
        guard !pieces.isEmpty else { return (rows, nil) }
        return (rows, pieces.count == 1 ? pieces[0] : concatenate(pieces))
    }
}
