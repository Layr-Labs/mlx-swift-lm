import Foundation

/// Decode complete committed blocks together, preserving tokenizer cleanup
/// inside a block. Hold trailing whitespace/incomplete UTF-8 until it is stable
/// across the next block; never repair or re-emit already-published bytes.
/// Unexpected non-boundary rewrites are recoverable errors, not corrupted JSON.
public final class CBv2NativeBlockTextDecoder {
    private let tokenizer: any Tokenizer
    private let holdback: StopHoldback
    private let stopScalars: [[Unicode.Scalar]]
    private var tokens = [Int]()
    private var emitted = [UInt8]()
    public private(set) var matchedStopString = false
    /// Original token prefix through the token completing the selected stop.
    /// A final token can contain additional bytes; tokens cannot be split or
    /// recovered by re-encoding the displayed prefix.
    public private(set) var stopTokenCount: Int?

    public init(tokenizer: any Tokenizer, stopStrings: [String]) {
        self.tokenizer = tokenizer
        self.holdback = StopHoldback(stopStrings: stopStrings)
        self.stopScalars = stopStrings.filter { !$0.isEmpty }.map { Array($0.unicodeScalars) }
    }

    public func append(_ block: [Int], terminal: Bool = false) throws -> String {
        guard !matchedStopString else { return "" }
        tokens.append(contentsOf: block)
        let decoded = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
        var scalars = Array(decoded.unicodeScalars)
        if !terminal {
            while let last = scalars.last,
                last.value == 0xfffd || last.properties.isWhitespace
            {
                scalars.removeLast()
            }
        }
        let bytes = Array(String(String.UnicodeScalarView(scalars)).utf8)
        guard bytes.starts(with: emitted) else {
            throw CBv2NativeBlockError.tokenizerRewroteCommittedText
        }
        let delta = String(decoding: bytes.dropFirst(emitted.count), as: UTF8.self)
        emitted = bytes
        let scan = holdback.ingest(delta)
        matchedStopString = scan.stopped
        if scan.stopped { stopTokenCount = try tokenBoundaryThroughStop(in: scalars) }
        return scan.text + (terminal && !scan.stopped ? holdback.flush() : "")
    }

    private func tokenBoundaryThroughStop(in scalars: [Unicode.Scalar]) throws -> Int {
        // Match the holdback's exact scalar semantics, not Foundation's
        // canonically equivalent string search. Earliest start wins; for
        // overlapping stops at that start, use the first completed delimiter.
        var selected: (start: Int, end: Int)?
        for stop in stopScalars where stop.count <= scalars.count {
            for start in 0...(scalars.count - stop.count) {
                guard scalars[start..<(start + stop.count)].elementsEqual(stop) else { continue }
                let end = start + stop.count
                if selected == nil || start < selected!.start
                    || (start == selected!.start && end < selected!.end) {
                    selected = (start, end)
                }
                break
            }
        }
        guard let selected, !tokens.isEmpty else {
            throw CBv2NativeBlockError.unsupportedRequest("stop boundary")
        }
        let prefix = Array(String(String.UnicodeScalarView(scalars[..<selected.end])).utf8)
        // Stop-only slow path. Prefix decoding respects cleanup, multi-byte
        // tokens and delimiters inside one token. Do not binary-search a
        // tokenizer whose cleanup is not necessarily monotonic.
        for count in 1...tokens.count {
            let decoded = tokenizer.decode(tokenIds: Array(tokens.prefix(count)), skipSpecialTokens: false)
            if decoded.utf8.starts(with: prefix) { return count }
        }
        throw CBv2NativeBlockError.unsupportedRequest("stop token boundary")
    }
}
