// Copyright © 2026 Eigen Labs.

import Foundation

/// Reads ATEM invokes the way the reference servers return them, for a caller
/// that splits Onyx channels itself and validates calls itself.
///
/// - Every complete `<atem:invoke>` element is one call, in order.
/// - A doubled name `X.X` is the declared tool `X`. Any other undeclared name
///   is passed on as generated, as vLLM does; `Call.isDeclared` says which.
/// - Values are typed as `OnyxReferenceDecoder` types them: a parameter the
///   schema types as a string keeps its text, a local `$ref` is resolved,
///   and `6.0` is an integer where only an integer is admitted.
///
/// Nothing here checks a call against its schema.
public struct ATEMReferenceReader: Sendable {
    public struct Call: Sendable, Equatable {
        public var toolCall: ToolCall
        /// The generated name is a declared tool, or its doubled form.
        public var isDeclared: Bool
    }

    public struct Reading: Sendable, Equatable {
        public var calls: [Call] = []
        /// Invokes that name no tool.
        public var unnamedInvokes = 0
        /// The text ends inside an invoke: its close, or a value's, is missing.
        public var endsInsideInvoke = false
    }

    private let tools: ATEMDeclaredTools
    private let toolSpecs: [[String: any Sendable]]?

    public init(tools: [[String: any Sendable]]?) {
        self.tools = ATEMDeclaredTools(tools)
        self.toolSpecs = tools
    }

    /// The calls in `text`: one tool channel's body, or one invoke element.
    public func read(_ text: String) -> Reading {
        let scan = ATEMReferenceScanner.scan(text)
        var reading = Reading(endsInsideInvoke: scan.endsInsideInvoke)
        for invoke in scan.invokes {
            guard let generated = invoke.name, !generated.isEmpty else {
                reading.unnamedInvokes += 1
                continue
            }
            let declared = tools.resolve(generated)
            let name = declared ?? generated
            var arguments: [String: JSONValue] = [:]
            for parameter in invoke.parameters {
                arguments[parameter.name] = tools.value(
                    parameter.value, parameter: parameter.name, ofTool: name)
            }
            let call = ToolArgumentNormalization.normalize(
                ToolCall(function: .init(name: name, arguments: arguments)), tools: toolSpecs)
            reading.calls.append(
                Call(
                    toolCall: ToolCall(
                        function: call.function, id: ToolCallFormat.atem.generateToolCallID()),
                    isDeclared: declared != nil))
        }
        return reading
    }
}
