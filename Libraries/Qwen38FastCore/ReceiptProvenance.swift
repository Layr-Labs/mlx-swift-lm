import CryptoKit
import Foundation

public struct Qwen38ReceiptArtifact: Codable, Equatable, Sendable {
    public let repository: String
    public let revision: String
    public let configSHA256: String

    public init(repository: String, revision: String, configSHA256: String) {
        self.repository = repository
        self.revision = revision
        self.configSHA256 = configSHA256
    }
}

public enum Qwen38TokenDigest {
    public static func sha256(_ tokens: [Int]) -> String {
        let data = Data(tokens.map(String.init).joined(separator: ",").utf8)
        return SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
