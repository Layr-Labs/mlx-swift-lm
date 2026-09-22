import MLXLMCommon
import Testing

@Suite("DiffusionGemma template syntax compatibility")
struct DiffusionGemmaTemplateCompatibilityTests {
    @Test func commentTrimAndBraceRulesMatchExistingProviderContract() {
        #expect(
            SwiftJinjaSyntaxCompatibility.normalize("before \n{#- hidden -#}\n after")
                == "before{# hidden #}after")
        #expect(
            SwiftJinjaSyntaxCompatibility.normalize("{ \n{{- value -}}")
                == "{{ '{' -}}{{- value -}}")
        #expect(
            SwiftJinjaSyntaxCompatibility.normalize("{ \n{%- if x -%}") == "{{ '{' -}}{%- if x -%}")
    }
    @Test func normalizationIsIdempotentAndLeavesRequestClockOwnershipAlone() {
        let source = "{{ bos_token }}\n{#- note -#}\n{ \n{{- value -}}{{ strftime_now('%Y') }}"
        let normalized = SwiftJinjaSyntaxCompatibility.normalize(source)
        #expect(SwiftJinjaSyntaxCompatibility.normalize(normalized) == normalized)
        #expect(normalized.hasSuffix("{{ strftime_now('%Y') }}"))
        #expect(!normalized.contains("set strftime_now"))
    }
}
