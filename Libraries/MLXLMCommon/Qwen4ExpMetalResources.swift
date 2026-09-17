import Foundation

/// Qwen sources share the paged kernel's sealed-app boundary. SwiftPM's
/// generated Bundle.module accessor does not understand our app packaging.
enum Qwen4ExpMetalResources {
    static let bundleName = "mlx-swift-lm_MLXLMCommon.bundle"
    static let names = ["gemm", "quantized_utils", "quantized"]

    enum ResourceError: Error, CustomStringConvertible {
        case missing(String)
        case outsideApp(String)
        case conflicting(String)

        var description: String {
            switch self {
            case .missing(let name):
                return "Required Qwen4 Metal resource is missing or empty: \(name)"
            case .outsideApp(let name):
                return "Qwen4 Metal resource escapes the signed app: \(name)"
            case .conflicting(let name):
                return "Conflicting Qwen4 Metal resources: \(name)"
            }
        }
    }

    static func load(
        _ name: String,
        executableURL: URL? = Bundle.main.executableURL,
        developmentSearchRoots: [URL]? = nil
    ) throws -> String {
        precondition(names.contains(name))
        let sealed = PagedAttentionResources.packagedAppResourcesURL(executableURL: executableURL)
        let roots = sealed.map { [$0] }
            ?? developmentSearchRoots
            ?? PagedAttentionResources.developmentRoots(executableURL: executableURL)
        var source: String?
        var visited = Set<String>()
        for root in roots {
            let bundle = root.lastPathComponent == bundleName
                ? root : root.appendingPathComponent(bundleName, isDirectory: true)
            let file = bundle.appendingPathComponent("Qwen4Metal/\(name).metal")
                .resolvingSymlinksInPath().standardizedFileURL
            if let sealed, !file.path.hasPrefix(
                sealed.standardizedFileURL.path + "/") {
                throw ResourceError.outsideApp(name)
            }
            guard visited.insert(file.path).inserted,
                let candidate = try? String(contentsOf: file, encoding: .utf8),
                !candidate.isEmpty else { continue }
            if let source, source != candidate { throw ResourceError.conflicting(name) }
            source = candidate
        }
        guard let source else { throw ResourceError.missing(name) }
        return source
    }
}
