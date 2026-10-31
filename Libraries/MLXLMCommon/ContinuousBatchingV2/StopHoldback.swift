// StopHoldback.swift
//
// Streaming stop-string matcher for ContinuousBatchingV2 (workstream E).
// One instance per in-flight request, fed the incremental text produced by
// `DetokenizerV2`.
//
// Guarantee: `ingest` only ever returns text that provably cannot be part
// of a stop string — the longest suffix of the pending text that is a
// prefix of ANY stop string is held back until it is either completed
// (stop: truncate at the match start, signal stop, emit nothing further)
// or disambiguated (released with the next chunk). This makes workstream
// B's one-step-late stop detection invisible: a user can never observe
// text at or past a stop match, even when the stop string spans several
// tokens or overlaps another candidate ("</s>" vs "</section>").
//
// `flush()` releases the held tail at natural end of generation (EOS /
// max-tokens), where no stop can complete anymore.
//
// Matching is over unicode scalars (exact substring semantics, no
// normalization), mirroring how OpenAI-compatible servers compare stop
// strings against the decoded stream.

import Foundation

public struct CBv2StopScan: Sendable, Equatable {
    /// Text that is safe to emit now.
    public var text: String
    /// True when a stop string fully matched. All text at and after the
    /// match start has been discarded; every further `ingest`/`flush`
    /// yields nothing.
    public var stopped: Bool

    public init(text: String, stopped: Bool) {
        self.text = text
        self.stopped = stopped
    }
}

/// Where stop strings apply in generated text, for reasoning models that
/// stream private reasoning, tool-call frames and the answer as one text.
///
/// - `init(opener:openerEnd:closers:)`: only inside answer spans, such as the
///   Harmony final message. `opener`, then `openerEnd` when set (the end of a
///   header that may carry attributes), opens a span; the first `closers`
///   entry closes it.
/// - `init(excluding:startingInside:)`: everywhere except inside the given
///   spans, their markers included, such as `<think>` reasoning and tool-call
///   frames. Whitespace that begins a run of matchable text cannot match on
///   its own: templates write it as a separator after reasoning, and collected
///   answers are trimmed.
///
/// Stops never match across a span boundary. Markers are expected to arrive
/// whole, as special tokens do. A stop that ends inside a marker split across
/// chunks matches before the marker is seen.
public struct CBv2StopStringScope: Sendable, Equatable {
    /// Text from `opener` through the first `closers` entry.
    public struct Span: Sendable, Equatable {
        public var opener: String
        public var closers: [String]

        public init(opener: String, closers: [String]) {
            self.opener = opener
            self.closers = closers
        }
    }

    public enum Matching: Sendable, Equatable {
        /// Only inside the span, after `headerEnd` when set.
        case inside(Span, headerEnd: String?)
        /// Only outside the spans. `startsInside`: the prompt already opened
        /// the first span, so output begins inside it.
        case outside([Span], startsInside: Bool)
    }

    public var matching: Matching

    public init(opener: String, openerEnd: String? = nil, closers: [String]) {
        self.matching = .inside(Span(opener: opener, closers: closers), headerEnd: openerEnd)
    }

    public init(excluding spans: [Span], startingInside: Bool = false) {
        self.matching = .outside(spans, startsInside: startingInside)
    }
}

public final class StopHoldback {

    private struct ScopeMarkers {
        let opener: [Unicode.Scalar]
        let openerEnd: [Unicode.Scalar]?
        let closers: [[Unicode.Scalar]]
    }

    private struct ExcludedSpans {
        let openers: [[Unicode.Scalar]]
        /// Per span, parallel to `openers`.
        let closers: [[[Unicode.Scalar]]]
    }

    private enum ScopePhase {
        case outside
        case header
        case inside
    }

    private let stops: [[Unicode.Scalar]]
    private let scope: ScopeMarkers?
    private var phase: ScopePhase = .outside
    private let excluded: ExcludedSpans?
    /// The excluded span the text is in, if any.
    private var excludedSpan: Int?
    /// The current run of matchable text has released a non-whitespace scalar.
    private var runHasText = false
    /// Held-back tail: the longest suffix of everything ingested so far
    /// that is a prefix of at least one stop string (or, when scoped, of the
    /// marker the current phase is waiting for).
    private var pending: [Unicode.Scalar] = []
    private var stopped = false

    public init(stopStrings: [String], scope: CBv2StopStringScope? = nil) {
        func markers(_ strings: [String]) -> [[Unicode.Scalar]] {
            strings.filter { !$0.isEmpty }.map { Array($0.unicodeScalars) }
        }
        self.stops = markers(stopStrings)
        switch scope?.matching {
        case .inside(let span, let headerEnd)? where !span.opener.isEmpty:
            self.scope = ScopeMarkers(
                opener: Array(span.opener.unicodeScalars),
                openerEnd: headerEnd.flatMap { $0.isEmpty ? nil : Array($0.unicodeScalars) },
                closers: markers(span.closers))
            self.excluded = nil
        case .outside(let spans, let startsInside)? where !spans.isEmpty:
            self.scope = nil
            self.excluded = ExcludedSpans(
                openers: spans.map { Array($0.opener.unicodeScalars) },
                closers: spans.map { markers($0.closers) })
            self.excludedSpan = startsInside ? 0 : nil
        default:
            self.scope = nil
            self.excluded = nil
        }
    }

    /// True when no stop strings are configured (pass-through mode).
    public var isPassthrough: Bool { stops.isEmpty }

    /// Feed newly detokenized text; returns what may be emitted.
    public func ingest(_ text: String) -> CBv2StopScan {
        if stopped {
            return CBv2StopScan(text: "", stopped: true)
        }
        if stops.isEmpty {
            return CBv2StopScan(text: text, stopped: false)
        }

        pending.append(contentsOf: text.unicodeScalars)
        if let scope {
            return ingestScoped(scope)
        }
        if let excluded {
            return ingestExcluding(excluded)
        }

        // 1) Earliest full match across all stop strings wins; everything
        //    from the match start on is discarded.
        if let matchStart = firstOccurrence(of: stops)?.start {
            stopped = true
            let emit = String(String.UnicodeScalarView(pending[..<matchStart]))
            pending.removeAll()
            return CBv2StopScan(text: emit, stopped: true)
        }

        // 2) No full match: hold the longest suffix that could still grow
        //    into a stop string; emit the rest.
        let hold = longestSuffixPrefixing(stops)
        let emitCount = pending.count - hold
        guard emitCount > 0 else {
            return CBv2StopScan(text: "", stopped: false)
        }
        let emit = String(String.UnicodeScalarView(pending[..<emitCount]))
        pending.removeFirst(emitCount)
        return CBv2StopScan(text: emit, stopped: false)
    }

    /// Release held text at natural end of generation (no stop matched).
    public func flush() -> String {
        guard !stopped, !pending.isEmpty else { return "" }
        let emit = String(String.UnicodeScalarView(pending))
        pending.removeAll()
        return emit
    }

    // MARK: - Internals

    /// Matches stops only inside spans and releases text up to each span
    /// boundary. A tail that could still grow into the next marker or a stop
    /// stays held.
    private func ingestScoped(_ scope: ScopeMarkers) -> CBv2StopScan {
        var released: [Unicode.Scalar] = []
        func release(_ count: Int) {
            released.append(contentsOf: pending[..<count])
            pending.removeFirst(count)
        }
        while true {
            switch phase {
            case .outside:
                if let opener = firstOccurrence(of: [scope.opener]) {
                    release(opener.start + opener.length)
                    phase = .header
                    continue
                }
                release(pending.count - longestSuffixPrefixing([scope.opener]))
            case .header:
                guard let openerEnd = scope.openerEnd else {
                    phase = .inside
                    continue
                }
                // A header that closes before it opens a span (malformed
                // output) returns to scanning for the next opener.
                let end = firstOccurrence(of: [openerEnd])
                let closer = firstOccurrence(of: scope.closers)
                if let closer, closer.start < end?.start ?? Int.max {
                    release(closer.start + closer.length)
                    phase = .outside
                    continue
                }
                if let end {
                    release(end.start + end.length)
                    phase = .inside
                    continue
                }
                release(pending.count - longestSuffixPrefixing([openerEnd] + scope.closers))
            case .inside:
                let closer = firstOccurrence(of: scope.closers)
                if let match = firstOccurrence(of: stops, endingBy: closer?.start) {
                    stopped = true
                    released.append(contentsOf: pending[..<match.start])
                    pending.removeAll()
                    return CBv2StopScan(
                        text: String(String.UnicodeScalarView(released)), stopped: true)
                }
                if let closer {
                    release(closer.start + closer.length)
                    phase = .outside
                    continue
                }
                release(pending.count - longestSuffixPrefixing(stops + scope.closers))
            }
            return CBv2StopScan(text: String(String.UnicodeScalarView(released)), stopped: false)
        }
    }

    /// Matches stops everywhere except inside excluded spans, markers
    /// included, and releases text up to each span boundary. A tail that
    /// could still grow into the next marker or a stop stays held.
    private func ingestExcluding(_ excluded: ExcludedSpans) -> CBv2StopScan {
        var released: [Unicode.Scalar] = []
        func release(_ count: Int) {
            released.append(contentsOf: pending[..<count])
            pending.removeFirst(count)
        }
        while true {
            if let span = excludedSpan {
                let closers = excluded.closers[span]
                if let closer = firstOccurrence(of: closers) {
                    release(closer.start + closer.length)
                    excludedSpan = nil
                    runHasText = false
                    continue
                }
                release(pending.count - longestSuffixPrefixing(closers))
            } else {
                let opener = firstOccurrence(of: excluded.openers)
                let end = opener?.start ?? pending.count
                // Until the run's first non-whitespace scalar, a stop matching
                // only whitespace (the separator before an answer) is ignored.
                let textStart =
                    runHasText ? 0 : pending[..<end].firstIndex { !$0.properties.isWhitespace }
                if let textStart,
                    let match = firstOccurrence(of: stops, endingBy: end, endingAfter: textStart)
                {
                    stopped = true
                    released.append(contentsOf: pending[..<match.start])
                    pending.removeAll()
                    return CBv2StopScan(
                        text: String(String.UnicodeScalarView(released)), stopped: true)
                }
                if let opener {
                    release(opener.start + opener.length)
                    excludedSpan = opener.index
                    continue
                }
                let count = pending.count - longestSuffixPrefixing(stops + excluded.openers)
                if let textStart, textStart < count {
                    runHasText = true
                }
                release(count)
            }
            return CBv2StopScan(text: String(String.UnicodeScalarView(released)), stopped: false)
        }
    }

    /// Earliest start (ties: longest) of any pattern that ends by `limit`,
    /// default the end of `pending`, and after `floor`. `pending` holds only
    /// the held tail plus the new chunk, so a full scan stays
    /// O(|pending|·Σ|pattern|).
    private func firstOccurrence(
        of patterns: [[Unicode.Scalar]], endingBy limit: Int? = nil, endingAfter floor: Int = 0
    ) -> (start: Int, length: Int, index: Int)? {
        let limit = limit ?? pending.count
        var found: (start: Int, length: Int, index: Int)?
        for (index, pattern) in patterns.enumerated()
        where !pattern.isEmpty && pattern.count <= limit {
            var start = max(0, floor - pattern.count + 1)
            while start <= limit - pattern.count {
                if let found, start > found.start { break }
                if pending[start ..< start + pattern.count].elementsEqual(pattern) {
                    if found == nil || start < found!.start
                        || (start == found!.start && pattern.count > found!.length)
                    {
                        found = (start, pattern.count, index)
                    }
                    break
                }
                start += 1
            }
        }
        return found
    }

    /// Length of the longest suffix of `pending` that is a proper prefix of
    /// at least one pattern (the part that must be held back).
    private func longestSuffixPrefixing(_ patterns: [[Unicode.Scalar]]) -> Int {
        var longest = 0
        for pattern in patterns {
            var k = min(pattern.count - 1, pending.count)
            while k > longest {
                if pending[(pending.count - k)...].elementsEqual(pattern[..<k]) {
                    longest = k
                    break
                }
                k -= 1
            }
        }
        return longest
    }
}
