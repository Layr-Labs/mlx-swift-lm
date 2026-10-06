/// Immutable identities for the two separately qualified native Qwen4 targets.
/// A profile selects test input identity, never an output golden or tolerance.
enum Qwen4RealArtifactProfile: String, CaseIterable {
    case affineQ4 = "affine-q4"
    case selectedQ4 = "selected-q4"

    static let environmentKey = "DARKBLOOM_QWEN4_REAL_ARTIFACT_PROFILE"

    static func requested(environment: [String: String]) throws -> Self {
        guard let raw = environment[environmentKey] else { return .affineQ4 }
        guard let profile = Self(rawValue: raw) else { throw Failure.unknownProfile }
        return profile
    }

    var configSHA256: String {
        switch self {
        case .affineQ4: "f04c5e8f500fe880617bf10a4ac6efc063b6ef8dbe66fb6a4843407c18ce7dae"
        case .selectedQ4: "319b334a1abb705acf06035aa0331bcb7c25976ff93a5035b543548738a10824"
        }
    }

    var indexSHA256: String {
        switch self {
        case .affineQ4: "7fccee8cf9d3b2a7af16e34dec85ccd3558bb3aa7a399ab9eee2b3211cd8fbde"
        case .selectedQ4: "05f70b017f328d7d9f955186bd9868d1f5c3fb708df89cd47d7d1c14b73b73d2"
        }
    }

    var tensorCount: Int { self == .affineQ4 ? 3866 : 3748 }
    var shardCount: Int { self == .affineQ4 ? 131 : 21 }
    var mtpTensorCount: Int { self == .affineQ4 ? 75 : 76 }

    func matches(configSHA256: String, indexSHA256: String,
                 tensorCount: Int, shardCount: Int, mtpTensorCount: Int) -> Bool {
        configSHA256 == self.configSHA256 && indexSHA256 == self.indexSHA256
            && tensorCount == self.tensorCount && shardCount == self.shardCount
            && mtpTensorCount == self.mtpTensorCount
    }

    enum Failure: Error { case unknownProfile }
}
