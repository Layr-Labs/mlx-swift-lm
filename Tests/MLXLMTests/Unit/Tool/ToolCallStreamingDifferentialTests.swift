import Foundation
import MLXLMCommon
import Testing

extension UnitTests {

    /// Seeded differential check of calls exposed while they are generated.
    /// Calls are rendered the way each chat template renders them (Gemma
    /// `format_argument`, Qwen 3.5 XML parameters, Hermes JSON, Harmony), with
    /// deviations that unconstrained `auto` output shows, and fed in token-like
    /// chunks: special tokens stay whole, other text splits at any Unicode
    /// scalar, and a chunk may carry several tokens (multi-token decode
    /// steps). Whatever the complete-frame path makes of the text, the
    /// streamed output must agree: the same calls in the same order, argument
    /// JSON that decodes to the arguments the whole-call path sends, exactly
    /// the text outside the frames, and a failure only for a frame the
    /// complete path also rejected.
    @Suite
    struct ToolCallStreamingDifferentialTests {

        enum Format: CaseIterable, Sendable {
            case gemma, qwenXML, qwenJSON, json, harmony

            var toolFormat: ToolCallFormat {
                switch self {
                case .gemma: .gemma
                case .qwenXML, .qwenJSON: .qwen35
                case .json: .json
                case .harmony: .harmony
                }
            }

            /// Sequences a model cannot place inside a value of this format
            /// without ending or re-framing the call.
            var reserved: [String] {
                switch self {
                case .gemma:
                    [
                        "<|\"|>", "<escape>", "<|tool_call>", "<tool_call|>",
                        "<start_function_call>", "<end_function_call>",
                    ]
                case .qwenXML: ["</parameter>", "<tool_call>", "</tool_call>", "</function>"]
                case .qwenJSON, .json: ["<tool_call>", "</tool_call>"]
                case .harmony: ["<|call|>", "<|end|>", "<|return|>"]
                }
            }

            /// Vocabulary tokens a decode step never splits.
            var specialTokens: [String] {
                switch self {
                case .gemma: ["<|tool_call>", "<tool_call|>", "<|\"|>"]
                case .qwenXML, .qwenJSON, .json: ["<tool_call>", "</tool_call>"]
                case .harmony:
                    [
                        "<|start|>", "<|channel|>", "<|message|>", "<|constrain|>", "<|call|>",
                        "<|end|>", "<|return|>",
                    ]
                }
            }

            var seedSalt: UInt64 { UInt64(Format.allCases.firstIndex(of: self)!) << 32 }
        }

        @Test(arguments: Format.allCases)
        func streamedCallsAgreeWithTheCompleteParse(_ format: Format) {
            for seed in UInt64(1) ... 500 {
                Self.check(seed: seed | format.seedSalt, format: format)
            }
        }

        // MARK: - Generation

        private struct Generator {
            private var state: UInt64

            init(seed: UInt64) { state = seed }

            /// SplitMix64.
            mutating func next() -> UInt64 {
                state &+= 0x9E37_79B9_7F4A_7C15
                var z = state
                z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
                z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
                return z ^ (z >> 31)
            }

            mutating func int(_ range: ClosedRange<Int>) -> Int {
                range.lowerBound + Int(next() % UInt64(range.count))
            }

            mutating func chance(_ percent: Int) -> Bool { int(0 ... 99) < percent }

            mutating func pick<T>(_ items: [T]) -> T { items[int(0 ... items.count - 1)] }

            mutating func shuffled<T>(_ items: [T]) -> [T] {
                var items = items
                for index in stride(from: items.count - 1, to: 0, by: -1) {
                    items.swapAt(index, int(0 ... index))
                }
                return items
            }
        }

        private enum Kind: String, CaseIterable {
            case string, integer, number, boolean, object, array
        }

        private indirect enum Value {
            case string(String)
            case integer(Int)
            case number(Double)
            case boolean(Bool)
            case object([(String, Value)])
            case array([Value])
        }

        private struct Call {
            var name: String
            var parameters: [(key: String, kind: Kind, value: Value)]
        }

        private static let pieces = [
            "a", "Paris", "hello world", " ", "  ", "\n", "\n\n", "\t", "\r\n", ",", ", ", ":",
            ": ", "{", "}", "[", "]", "\"", "\\", "\\n", "'", "é", "漢字", "🙂", "👨‍👩‍👧", "e\u{301}",
            "\u{1}", "\u{7f}", "\u{2028}", "x=1", "<", ">", "</", "<b>", "&amp;", "-", "_", "0",
            "3.14", "true", "null", "call:", "functions.", "<|", "|>", "<tool", "</param",
            "<parameter=k>", "<function=f>", "<|\"", "<escape", "def f():\n    return 1\n",
            "SELECT * FROM t;", "a,b:c", "</tool_call", "<|message|>", "<|channel|>",
            "to=functions.x",
        ]

        private static func string(_ generator: inout Generator, format: Format) -> String {
            while true {
                var text = ""
                for _ in 0 ..< generator.int(0 ... 6) { text += generator.pick(pieces) }
                if !format.reserved.contains(where: { text.contains($0) }) { return text }
            }
        }

        private static func value(
            _ kind: Kind, _ generator: inout Generator, format: Format, depth: Int
        ) -> Value {
            switch kind {
            case .string: return .string(string(&generator, format: format))
            case .integer: return .integer(generator.pick([0, 1, -1, 7, 42, 420, -3000, 100_000]))
            case .number: return .number(Double(generator.int(-400 ... 400) * 2 + 1) / 4)
            case .boolean: return .boolean(generator.chance(50))
            case .object:
                var members: [(String, Value)] = []
                var keys = Set<String>()
                for _ in 0 ..< generator.int(0 ... 3) {
                    let key = generator.pick(["id", "name", "path", "n", "tags", "owner", "x_y"])
                    guard keys.insert(key).inserted else { continue }
                    members.append((key, nested(&generator, format: format, depth: depth)))
                }
                return .object(members)
            case .array:
                return .array(
                    (0 ..< generator.int(0 ... 3)).map { _ in
                        nested(&generator, format: format, depth: depth)
                    })
            }
        }

        private static func nested(_ generator: inout Generator, format: Format, depth: Int)
            -> Value
        {
            let kinds: [Kind] =
                depth >= 2 ? [.string, .integer, .number, .boolean] : Kind.allCases
            return value(generator.pick(kinds), &generator, format: format, depth: depth + 1)
        }

        private static func call(_ generator: inout Generator, format: Format, name: String) -> Call
        {
            var parameters: [(key: String, kind: Kind, value: Value)] = []
            for key in ["path", "content", "count", "ratio", "force", "meta", "items"]
            where generator.chance(45) {
                let kind = generator.pick(Kind.allCases)
                parameters.append((key, kind, value(kind, &generator, format: format, depth: 0)))
            }
            return Call(name: name, parameters: generator.shuffled(parameters))
        }

        /// One declared tool per distinct call name, typed like its arguments.
        private static func tools(_ calls: [Call]) -> [[String: any Sendable]] {
            var properties: [String: [String: any Sendable]] = [:]
            var order: [String] = []
            for call in calls {
                if properties[call.name] == nil { order.append(call.name) }
                for parameter in call.parameters {
                    properties[call.name, default: [:]][parameter.key] =
                        ["type": parameter.kind.rawValue] as [String: any Sendable]
                }
            }
            var tools: [[String: any Sendable]] = order.map { name in
                [
                    "type": "function",
                    "function": [
                        "name": name,
                        "parameters": ["type": "object", "properties": properties[name] ?? [:]]
                            as [String: any Sendable],
                    ] as [String: any Sendable],
                ]
            }
            tools.append([
                "type": "function",
                "function": ["name": "unused_tool", "parameters": ["type": "object"]]
                    as [String: any Sendable],
            ])
            return tools
        }

        // MARK: - Rendering

        private static func json(_ value: Value, _ generator: inout Generator, pretty: Bool)
            -> String
        {
            switch value {
            case .string(let text): return quoted(text, &generator)
            case .integer(let number): return String(number)
            case .number(let number): return String(number)
            case .boolean(let flag): return flag ? "true" : "false"
            case .array(let items):
                return "["
                    + items.map { json($0, &generator, pretty: pretty) }
                    .joined(separator: pretty ? ", " : ",") + "]"
            case .object(let members):
                return "{"
                    + members.map {
                        quoted($0.0, &generator) + (pretty ? ": " : ":")
                            + json($0.1, &generator, pretty: pretty)
                    }.joined(separator: pretty ? ", " : ",") + "}"
            }
        }

        /// A valid JSON string literal; escaping choices vary as models' do.
        private static func quoted(_ text: String, _ generator: inout Generator) -> String {
            var out = "\""
            for scalar in text.unicodeScalars {
                switch scalar {
                case "\"": out += "\\\""
                case "\\": out += "\\\\"
                case "\n": out += "\\n"
                case "\r": out += "\\r"
                case "\t": out += "\\t"
                case "/" where generator.chance(30): out += "\\/"
                default:
                    if scalar.value < 0x20 || (scalar.value > 0x7e && generator.chance(20)) {
                        if scalar.value > 0xFFFF {
                            let offset = scalar.value - 0x10000
                            out += String(
                                format: "\\u%04x\\u%04x", 0xD800 + (offset >> 10),
                                0xDC00 + (offset & 0x3FF))
                        } else {
                            out += String(format: "\\u%04x", scalar.value)
                        }
                    } else {
                        out.unicodeScalars.append(scalar)
                    }
                }
            }
            return out + "\""
        }

        /// Gemma `format_argument(value, escape_keys=False)`.
        private static func gemma(_ value: Value) -> String {
            switch value {
            case .string(let text): return "<|\"|>" + text + "<|\"|>"
            case .integer(let number): return String(number)
            case .number(let number): return String(number)
            case .boolean(let flag): return flag ? "true" : "false"
            case .array(let items): return "[" + items.map(gemma).joined(separator: ",") + "]"
            case .object(let members):
                return "{"
                    + members.sorted { $0.0 < $1.0 }.map { $0.0 + ":" + gemma($0.1) }
                    .joined(separator: ",") + "}"
            }
        }

        /// One frame; `closed` is false when the end tag is a stop token the
        /// text stream never shows.
        private static func render(
            _ call: Call, format: Format, _ generator: inout Generator, closed: Bool
        ) -> String {
            switch format {
            case .gemma:
                let separator = generator.chance(15) ? ", " : ","
                let members = call.parameters.sorted { $0.key < $1.key }.map {
                    parameter -> String in
                    // Unconstrained output occasionally drops a string's markers.
                    if case .string(let text) = parameter.value, generator.chance(10),
                        !text.isEmpty,
                        !text.contains(where: { ",:{}[]<\"\\".contains($0) || $0.isNewline })
                    {
                        return parameter.key + ":" + text
                    }
                    return parameter.key + (generator.chance(10) ? ": " : ":")
                        + gemma(parameter.value)
                }
                return "<|tool_call>call:\(call.name){" + members.joined(separator: separator)
                    + "}" + (closed ? "<tool_call|>" : "")
            case .qwenXML:
                var text = "<tool_call>\n<function=\(call.name)>\n"
                for parameter in call.parameters {
                    let rendered: String
                    switch parameter.value {
                    case .string(let raw): rendered = raw
                    case .integer(let number): rendered = String(number)
                    case .number(let number): rendered = String(number)
                    case .boolean(let flag): rendered = flag ? "true" : "false"
                    case .object, .array: rendered = json(parameter.value, &generator, pretty: true)
                    }
                    text += "<parameter=\(parameter.key)>\n\(rendered)\n</parameter>\n"
                }
                return text + "</function>\n" + (closed ? "</tool_call>" : "")
            case .qwenJSON, .json:
                let arguments = json(
                    .object(call.parameters.map { ($0.key, $0.value) }), &generator,
                    pretty: generator.chance(50))
                let name = quoted(call.name, &generator)
                let body =
                    generator.chance(15)
                    ? "{\"arguments\": \(arguments), \"name\": \(name)}"
                    : "{\"name\": \(name), \"arguments\": \(arguments)}"
                return "<tool_call>\n" + body + "\n" + (closed ? "</tool_call>" : "")
            case .harmony:
                let header =
                    generator.chance(50)
                    ? "<|start|>assistant<|channel|>commentary to=functions.\(call.name) <|constrain|>json<|message|>"
                    : "<|start|>assistant to=functions.\(call.name)<|channel|>commentary json<|message|>"
                let arguments = json(
                    .object(call.parameters.map { ($0.key, $0.value) }), &generator,
                    pretty: generator.chance(30))
                return header + arguments + (closed ? "<|call|>" : "")
            }
        }

        // MARK: - Running

        /// Token-like units (special tokens whole, other text in 1-4 scalar
        /// pieces) grouped 1-3 per chunk.
        private static func chunks(_ text: String, format: Format, _ generator: inout Generator)
            -> [String]
        {
            let scalars = Array(text.unicodeScalars)
            let specials = format.specialTokens.map { Array($0.unicodeScalars) }
            func special(at index: Int) -> Int? {
                specials.first { token in
                    index + token.count <= scalars.count
                        && Array(scalars[index ..< index + token.count]) == token
                }?.count
            }
            var units: [String] = []
            var index = 0
            while index < scalars.count {
                if let length = special(at: index) {
                    units.append(scalarString(scalars[index ..< index + length]))
                    index += length
                    continue
                }
                var end = index
                let limit = min(scalars.count, index + generator.int(1 ... 4))
                while end < limit, end == index || special(at: end) == nil { end += 1 }
                units.append(scalarString(scalars[index ..< end]))
                index = end
            }
            var out: [String] = []
            var start = 0
            while start < units.count {
                let end = min(units.count, start + generator.int(1 ... 3))
                out.append(units[start ..< end].joined())
                start = end
            }
            return out
        }

        private static func scalarString(_ scalars: ArraySlice<Unicode.Scalar>) -> String {
            var view = String.UnicodeScalarView()
            view.append(contentsOf: scalars)
            return String(view)
        }

        private struct Observed {
            var text = ""
            var names: [String] = []
            var arguments: [String] = []
            var completed: [ToolCall] = []
            var failures: [ToolCallStreamFailure] = []
        }

        private static func streamed(
            _ chunks: [String], format: ToolCallFormat, tools: [[String: any Sendable]]
        ) -> Observed {
            let processor = ToolCallProcessor(
                format: format, tools: tools, streaming: .init(exposesName: { _ in true }))
            var observed = Observed()
            var open: Int?
            let segments =
                chunks.flatMap { processor.processChunkSegments($0) }
                + processor.processEOSSegments()
            for segment in segments {
                switch segment {
                case .text(let piece):
                    observed.text += piece
                case .callStarted(let name):
                    observed.names.append(name)
                    observed.arguments.append("")
                    open = observed.arguments.count - 1
                case .callArguments(let fragment):
                    if let open { observed.arguments[open] += fragment }
                case .callCompleted(let call, let remainder):
                    if let open { observed.arguments[open] += remainder }
                    observed.completed.append(call)
                    open = nil
                case .call(let call):
                    observed.names.append(call.function.name)
                    observed.arguments.append(encoded(call.function.arguments))
                    observed.completed.append(call)
                case .callFailed(let failure):
                    observed.failures.append(failure)
                    open = nil
                }
            }
            #expect(processor.toolCalls.isEmpty)
            return observed
        }

        private static func encoded(_ arguments: [String: JSONValue]) -> String {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            return String(decoding: try! encoder.encode(arguments), as: UTF8.self)
        }

        /// JSON numbers compare by value: `1` and `1.0` are the same argument.
        private static func normalized(_ value: JSONValue) -> JSONValue {
            switch value {
            case .int(let number): .double(Double(number))
            case .array(let items): .array(items.map(normalized))
            case .object(let members): .object(members.mapValues(normalized))
            default: value
            }
        }

        private static func decoded(_ text: String) -> JSONValue? {
            (try? JSONDecoder().decode(JSONValue.self, from: Data(text.utf8))).map(normalized)
        }

        /// What the complete-frame path makes of one frame delivered whole:
        /// its call, or nothing when it falls back to text.
        private static func reference(
            _ frame: String, format: Format, tools: [[String: any Sendable]]
        ) -> [ToolCall] {
            let processor = ToolCallProcessor(format: format.toolFormat, tools: tools)
            _ = processor.processChunk(frame)
            _ = processor.processEOS(returnBufferedText: true)
            return processor.toolCalls
        }

        /// One generated response checked against the complete-frame path.
        /// The reference parses each frame delivered whole, so it does not
        /// inherit the non-streaming path's own chunk-boundary limits (it can
        /// miss a start tag that shares a chunk with earlier text).
        private static func check(seed: UInt64, format: Format) {
            var generator = Generator(seed: seed)
            let names = ["write_file", "get_weather", "run"]
            let calls = (0 ..< (generator.chance(25) ? 2 : 1)).map { _ in
                call(&generator, format: format, name: generator.pick(names))
            }
            let tools = tools(calls)
            var text = generator.chance(20) ? "Sure, calling it now.\n" : ""
            var outside = text
            var expected: [ToolCall] = []
            var rejectedFrames = 0
            for (index, call) in calls.enumerated() {
                // A trailing end tag that is a stop token never reaches the text.
                let closed = index < calls.count - 1 || generator.chance(60)
                let frame = render(call, format: format, &generator, closed: closed)
                let parsed = reference(frame, format: format, tools: tools)
                expected += parsed
                if parsed.isEmpty { rejectedFrames += 1 }
                text += frame
                if index < calls.count - 1 {
                    let separator = generator.pick(["", "\n", "\n\n"])
                    text += separator
                    outside += separator
                }
            }

            let pieces = chunks(text, format: format, &generator)
            let observed = streamed(pieces, format: format.toolFormat, tools: tools)

            let context = Comment(
                rawValue: "seed \(seed) \(format) chunks \(pieces.debugDescription)")
            if outside.isEmpty, calls.count == 1 {
                // A response that is one frame from its first token: the
                // complete-frame path itself is chunking-invariant.
                let legacy = ToolCallProcessor(format: format.toolFormat, tools: tools)
                var legacyText = ""
                for piece in pieces { legacyText += legacy.processChunk(piece) ?? "" }
                legacyText += legacy.processEOS(returnBufferedText: true) ?? ""
                #expect(legacy.toolCalls == expected, context)
                if !expected.isEmpty { #expect(legacyText.isEmpty, context) }
            }
            #expect(observed.completed == expected, context)
            // Only a frame the complete path also rejects may fail.
            #expect(observed.failures.count <= rejectedFrames, context)
            if observed.failures.isEmpty, rejectedFrames == 0 {
                #expect(observed.text == outside, context)
            }
            #expect(
                observed.names.count == observed.completed.count + observed.failures.count, context)
            var completedIndex = 0
            for (index, name) in observed.names.enumerated()
            where completedIndex < observed.completed.count {
                let call = observed.completed[completedIndex]
                guard name == call.function.name else { continue }
                completedIndex += 1
                #expect(
                    decoded(observed.arguments[index]) == decoded(encoded(call.function.arguments)),
                    context)
            }
        }
    }
}
