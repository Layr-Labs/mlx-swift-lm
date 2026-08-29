import CryptoKit
import Foundation

public enum Qwen38ArtifactValidationError: Error, CustomStringConvertible {
    case missing(String)
    case digest(path: String, expected: String, actual: String)

    public var description: String {
        switch self {
        case .missing(let path):
            "pinned artifact file is missing: \(path)"
        case .digest(let path, let expected, let actual):
            "pinned artifact digest mismatch for \(path): expected \(expected), got \(actual)"
        }
    }
}

public enum Qwen38ArtifactValidator {
    public static func validate(
        directory: URL,
        reference: Qwen38ArtifactReference
    ) throws -> String {
        let configURL = directory.appending(component: "config.json")
        let configSHA256 = try sha256(url: configURL)
        guard reference.configSHA256s.contains(configSHA256) else {
            throw Qwen38ArtifactValidationError.digest(
                path: configURL.path,
                expected: reference.configSHA256s.sorted().joined(separator: " or "),
                actual: configSHA256)
        }
        for (name, expected) in reference.requiredFileSHA256.sorted(by: { $0.key < $1.key }) {
            precondition(
                !name.contains("/") && !name.contains("\\"),
                "artifact manifest file names must be local")
            try validate(directory.appending(component: name), expected: expected)
        }
        return configSHA256
    }

    public static func sha256(url: URL) throws -> String {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Qwen38ArtifactValidationError.missing(url.path)
        }
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var hasher = SHA256()
        while let data = try file.read(upToCount: 4 * 1_024 * 1_024), !data.isEmpty {
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func validate(_ url: URL, expected: String) throws {
        let actual = try sha256(url: url)
        guard actual == expected else {
            throw Qwen38ArtifactValidationError.digest(
                path: url.path,
                expected: expected,
                actual: actual)
        }
    }
}
