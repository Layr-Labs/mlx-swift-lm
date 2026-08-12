// Copyright © 2026 Eigen Labs.

@preconcurrency import AVFoundation
import Foundation

/// Errors raised before or while AVFoundation reads an in-memory ISO BMFF video.
public enum MemoryBackedVideoAssetError: Error, Equatable, Sendable {
    case emptyData
    case invalidContainer
    case invalidRange
    case unexpectedResource
}

/// Owns an `AVURLAsset` whose bytes are served directly from memory.
///
/// `AVAssetResourceLoader` keeps its delegate weakly, so returning a bare
/// `AVURLAsset` is unsafe: the delegate (and therefore the bytes) can disappear
/// while AVFoundation is still probing or sampling the video. This owner keeps
/// the asset, loader, data, and delegate queue alive together. The custom URL is
/// never backed by a file or network resource.
public final class MemoryBackedVideoAsset: @unchecked Sendable {
    private let asset: AVURLAsset

    private let loader: MemoryAssetResourceLoader
    private let delegateQueue: DispatchQueue

    public var byteCount: Int { loader.byteCount }

    /// Creates an in-memory ISO BMFF MP4 or QuickTime asset.
    ///
    /// The first box must be a complete `ftyp` box. This intentionally rejects
    /// arbitrary bytes and non-ISO-BMFF container formats before AVFoundation
    /// sees them; callers must fail closed rather than letting the framework
    /// infer an unrelated format.
    public init(videoData data: Data) throws {
        let contentType = try Self.validatedContentType(data)

        let token = UUID().uuidString.lowercased()
        guard let resourceURL = URL(string: "darkbloom-memory-video://\(token)/asset.mp4") else {
            throw MemoryBackedVideoAssetError.invalidContainer
        }
        let loader = MemoryAssetResourceLoader(
            data: data,
            resourceURL: resourceURL,
            contentType: contentType)
        let queue = DispatchQueue(
            label: "ai.darkbloom.memory-video.\(token)",
            qos: .userInitiated)
        let asset = AVURLAsset(
            url: resourceURL,
            options: [
                AVURLAssetPreferPreciseDurationAndTimingKey: true,
                AVURLAssetReferenceRestrictionsKey:
                    AVAssetReferenceRestrictions.forbidAll.rawValue,
            ])
        asset.resourceLoader.setDelegate(loader, queue: queue)

        self.asset = asset
        self.loader = loader
        self.delegateQueue = queue
    }

    /// Runs an operation while keeping the asset's resource-loader owner alive.
    ///
    /// This is package-scoped so a bare memory-backed asset cannot escape through
    /// the public API. The explicit lifetime guarantee covers every suspension
    /// and every throwing exit from the operation.
    package func withAsset<Result>(
        _ operation: (AVURLAsset) async throws -> Result
    ) async rethrows -> Result {
        defer { withExtendedLifetime(self) {} }
        return try await operation(asset)
    }

    package var resourceRequestCount: Int { loader.resourceRequestCount }

    static func validatedContentType(_ data: Data) throws -> String {
        // ISO BMFF: 32-bit size + "ftyp" + major brand + minor version.
        guard data.count >= 16 else {
            throw data.isEmpty
                ? MemoryBackedVideoAssetError.emptyData
                : MemoryBackedVideoAssetError.invalidContainer
        }
        let prefix = [UInt8](data.prefix(24))
        guard prefix[4] == 0x66, prefix[5] == 0x74, prefix[6] == 0x79, prefix[7] == 0x70
        else {
            throw MemoryBackedVideoAssetError.invalidContainer
        }

        let size32 = prefix[0 ... 3].reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
        let boxSize: UInt64
        let minimumSize: UInt64
        let majorBrandOffset: Int
        if size32 == 1 {
            guard data.count >= 24 else { throw MemoryBackedVideoAssetError.invalidContainer }
            boxSize = prefix[8 ... 15].reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
            minimumSize = 24
            majorBrandOffset = 16
        } else {
            // A zero size means "to EOF" in ISO BMFF. It is legal, but an
            // `ftyp` box that consumes the whole file cannot contain media, so
            // reject it here instead of relying on a later metadata failure.
            guard size32 != 0 else { throw MemoryBackedVideoAssetError.invalidContainer }
            boxSize = UInt64(size32)
            minimumSize = 16
            majorBrandOffset = 8
        }
        guard boxSize >= minimumSize, boxSize <= UInt64(data.count) else {
            throw MemoryBackedVideoAssetError.invalidContainer
        }
        return prefix[majorBrandOffset ..< majorBrandOffset + 4].elementsEqual(
            [0x71, 0x74, 0x20, 0x20])
            ? AVFileType.mov.rawValue : AVFileType.mp4.rawValue
    }

    /// Computes the safe slice for an AVFoundation data request.
    ///
    /// AVFoundation may ask past EOF; that request is clamped to EOF. Negative,
    /// overflowing, backward, and wholly out-of-resource offsets are rejected.
    /// Kept internal so range semantics can be tested without constructing the
    /// framework-owned `AVAssetResourceLoadingRequest` type.
    static func responseRange(
        resourceLength: Int,
        requestedOffset: Int64,
        requestedLength: Int,
        currentOffset: Int64,
        requestsAllDataToEnd: Bool
    ) throws -> Range<Int> {
        guard resourceLength >= 0, requestedOffset >= 0, requestedLength >= 0,
            currentOffset >= 0
        else { throw MemoryBackedVideoAssetError.invalidRange }

        let length = Int64(resourceLength)
        let start = max(requestedOffset, currentOffset)
        guard start <= length else { throw MemoryBackedVideoAssetError.invalidRange }

        let requestedEnd: Int64
        if requestsAllDataToEnd {
            requestedEnd = length
        } else {
            let (end, overflow) = requestedOffset.addingReportingOverflow(
                Int64(requestedLength))
            guard !overflow, end >= start else {
                throw MemoryBackedVideoAssetError.invalidRange
            }
            requestedEnd = min(end, length)
        }

        return Int(start) ..< Int(requestedEnd)
    }
}

private final class MemoryAssetResourceLoader: NSObject,
    AVAssetResourceLoaderDelegate, @unchecked Sendable
{
    let byteCount: Int

    var resourceRequestCount: Int {
        stateLock.withLock { _resourceRequestCount }
    }

    private let data: Data
    private let resourceURL: URL
    private let contentType: String
    private let stateLock = NSLock()
    private var _resourceRequestCount = 0

    init(data: Data, resourceURL: URL, contentType: String) {
        self.data = data
        self.byteCount = data.count
        self.resourceURL = resourceURL
        self.contentType = contentType
        super.init()
    }

    func resourceLoader(
        _ resourceLoader: AVAssetResourceLoader,
        shouldWaitForLoadingOfRequestedResource loadingRequest: AVAssetResourceLoadingRequest
    ) -> Bool {
        stateLock.withLock {
            _resourceRequestCount += 1
        }
        guard loadingRequest.request.url == resourceURL else {
            loadingRequest.finishLoading(with: MemoryBackedVideoAssetError.unexpectedResource)
            return true
        }

        if let information = loadingRequest.contentInformationRequest {
            information.contentType = contentType
            information.contentLength = Int64(data.count)
            information.isByteRangeAccessSupported = true
        }

        guard let request = loadingRequest.dataRequest else {
            loadingRequest.finishLoading()
            return true
        }

        do {
            let range = try MemoryBackedVideoAsset.responseRange(
                resourceLength: data.count,
                requestedOffset: request.requestedOffset,
                requestedLength: request.requestedLength,
                currentOffset: request.currentOffset,
                requestsAllDataToEnd: request.requestsAllDataToEndOfResource)
            if !range.isEmpty {
                let lowerBound = data.index(data.startIndex, offsetBy: range.lowerBound)
                let upperBound = data.index(data.startIndex, offsetBy: range.upperBound)
                request.respond(with: data[lowerBound ..< upperBound])
            }
            loadingRequest.finishLoading()
        } catch {
            loadingRequest.finishLoading(with: error)
        }
        return true
    }

    func resourceLoader(
        _ resourceLoader: AVAssetResourceLoader,
        didCancel loadingRequest: AVAssetResourceLoadingRequest
    ) {
        // Cancellation is terminal and owned by AVFoundation. In particular,
        // do not retry, fetch another URL, or materialize a fallback file.
    }
}
