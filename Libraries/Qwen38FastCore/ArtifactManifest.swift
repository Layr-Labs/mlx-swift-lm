public struct Qwen38ArtifactReference: Equatable, Sendable {
    public let repository: String
    public let revision: String
    public let configSHA256s: Set<String>
    public let requiredFileSHA256: [String: String]

    public init(
        repository: String,
        revision: String,
        configSHA256s: Set<String>,
        requiredFileSHA256: [String: String]
    ) {
        self.repository = repository
        self.revision = revision
        self.configSHA256s = configSHA256s
        self.requiredFileSHA256 = requiredFileSHA256
    }
}

public struct Qwen38ArtifactManifest: Equatable, Sendable {
    public let swiftBaseRevision: String
    public let mlxSwiftRevision: String
    public let mlxRevision: String
    public let mtplxSourceRevision: String
    public let yukonSourceRevision: String
    public let dflash2SourceRevision: String
    public let targetArtifact: Qwen38ArtifactReference
    public let draftArtifact: Qwen38ArtifactReference

    public static let production = Qwen38ArtifactManifest(
        swiftBaseRevision: "ab73a827c9dde6f8802507003aa0be71605aab8e",
        mlxSwiftRevision: "606d28cfa8c1d66b2975d3162a4aac9756835c5f",
        mlxRevision: "0a725e3000edabc4911cde345270ca950bfa152f",
        mtplxSourceRevision: "26e27b78d5299dcafb319844283ac50a137bfee5",
        yukonSourceRevision: "eb5eadc7a165047d4321ce883b9ff30894d8bd19",
        dflash2SourceRevision: "c5b76ddb62bdefb6eeef1282641842edcf23a1b8",
        targetArtifact: Qwen38ArtifactReference(
            repository: "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed",
            revision: "123db8bcc7101455b00d9aad36c0e760c6e7de02",
            configSHA256s: [
                "533e833dedb9e7b6a8ee22ab4f2fc034bcf6ded9d8693e5ebcc9d5f159b62a3b",
                "913b57d11eb131d9e1bb7316ea8729b6bcd110b59f8a08840a83ba2e524f370d",
            ],
            requiredFileSHA256: [
                "chat_template.jinja":
                    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
                "model-00001-of-00004.safetensors":
                    "aed435f4011667af7772fee7ccec90c86f5a6c85649ef74fe6e397a7504b4578",
                "model-00002-of-00004.safetensors":
                    "3f1960306e36255b7f0c7e80c742f03d96131561b766b1f6af9ca138ff77f939",
                "model-00003-of-00004.safetensors":
                    "6525c0edae616c0f62b69fc190268c69bdf7a18b0b2ad3de007f3e4464387471",
                "model-00004-of-00004.safetensors":
                    "cef192620e5ecb23eaac19d2c041edc44d9cbb9fb10265bf1213b649c3eae0d9",
                "model-vision.safetensors":
                    "964bef26c740bdb6fe464b4c7c48840d4c952ea597c181c41cb131c65ea3c5d5",
                "model.safetensors.index.json":
                    "aec76dd8123c07e947f22a0e391bc278c05989094d398a64d759b76a93e3754e",
                "mtp.safetensors":
                    "4468f39621de68a19ffd0bcb2e2e2f352205def7436a625b3427e3752866c287",
                "tokenizer.json":
                    "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
                "tokenizer_config.json":
                    "95c557768e6b88a7128befc7bfd3c7de50e5d51af9b8b33a9f4dee0e04f99679",
            ]),
        draftArtifact: Qwen38ArtifactReference(
            repository: "z-lab/Qwen3.8-27B-DFlash2",
            revision: "50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
            configSHA256s: [
                "873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980"
            ],
            requiredFileSHA256: [
                "model.safetensors":
                    "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c"
            ]))
}
