import MLXLMCommon
import Testing

@Suite("Qwen4 native tool wire format")
struct Qwen4ToolFormatTests {
    @Test func nativeArchitectureSelectsFramedDualDialectParser() throws {
        for name in ["qwen4_exp", "qwen4_exp_text"] {
            let format = try #require(ToolCallFormat.infer(from: name))
            #expect(format == .qwen35)
            let parser = format.createParser()
            for payload in [
                "<function=add><parameter=a>19</parameter><parameter=b>23</parameter></function>",
                #"{"name":"add","arguments":{"a":19,"b":23}}"#,
            ] {
                let call = try #require(parser.parse(content: payload, tools: nil))
                #expect(call.function.name == "add")
                #expect(call.function.arguments.count == 2)
            }
        }
        #expect(ToolCallFormat.infer(from: "nemotron_h") == .nemotron)
        #expect(ToolCallFormat.infer(from: "qwen3_next") == .xmlFunction)
    }
}
