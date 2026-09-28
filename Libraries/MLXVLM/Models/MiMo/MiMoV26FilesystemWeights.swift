// Copyright © 2026 Eigen Labs.
// MiMo-specific bounded local metadata gate over MLX.loadArrays(.cpu).
// Native arrays are lazy Load handles; this API never evaluates model payloads.
import CryptoKit
import Foundation
import MLX
import MLXLMCommon
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

public enum MiMoV26FilesystemError: Error, Equatable, Sendable {
    case cancelled, invalidRoot, unsafeFile(String), changedObject(String)
    case limit(String), invalidMetadata(String), invalidHeader(String), nativeLoad(String)
}

public struct MiMoV26FilesystemLimits: Sendable {
    public let maximumShardBytes, maximumTotalFileBytes: Int
    public let maximumConfigurationBytes, maximumIndexBytes, maximumHeaderBytes: Int
    public let maximumMetadataBytes, maximumShards, maximumTensors: Int
    public init(maximumShardBytes: Int, maximumTotalFileBytes: Int,
                maximumConfigurationBytes: Int = 1_048_576, maximumIndexBytes: Int = 4_194_304,
                maximumHeaderBytes: Int = 1_048_576, maximumMetadataBytes: Int = 16_777_216,
                maximumShards: Int = 64, maximumTensors: Int = 2048) {
        self.maximumShardBytes = maximumShardBytes; self.maximumTotalFileBytes = maximumTotalFileBytes
        self.maximumConfigurationBytes = maximumConfigurationBytes; self.maximumIndexBytes = maximumIndexBytes
        self.maximumHeaderBytes = maximumHeaderBytes; self.maximumMetadataBytes = maximumMetadataBytes
        self.maximumShards = maximumShards; self.maximumTensors = maximumTensors
    }
}

/// Exact local identity/state, not a content hash. All timestamps retain ns.
public struct MiMoV26FilesystemObjectState: Codable, Equatable, Sendable {
    public let device, inode: String
    public let bytes: Int
    public let modifiedSeconds, modifiedNanoseconds, changedSeconds, changedNanoseconds: Int64
    public init(device: String, inode: String, bytes: Int, modifiedSeconds: Int64,
                modifiedNanoseconds: Int64, changedSeconds: Int64, changedNanoseconds: Int64) {
        self.device = device; self.inode = inode; self.bytes = bytes
        self.modifiedSeconds = modifiedSeconds; self.modifiedNanoseconds = modifiedNanoseconds
        self.changedSeconds = changedSeconds; self.changedNanoseconds = changedNanoseconds
    }
}

public struct MiMoV26FilesystemExpectedFile: Codable, Sendable {
    public let objectState: MiMoV26FilesystemObjectState?
    public let headerSHA256: String?
    /// Prior immutable digest reference only: payload hashing is not performed.
    public let payloadSHA256: String?
    public init(objectState: MiMoV26FilesystemObjectState? = nil, headerSHA256: String? = nil, payloadSHA256: String? = nil) {
        self.objectState = objectState; self.headerSHA256 = headerSHA256; self.payloadSHA256 = payloadSHA256
    }
}
public struct MiMoV26FilesystemExpectations: Codable, Sendable {
    public let configurationSHA256, indexSHA256: String?
    public let files: [String: MiMoV26FilesystemExpectedFile]
    public init(configurationSHA256: String? = nil, indexSHA256: String? = nil,
                files: [String: MiMoV26FilesystemExpectedFile] = [:]) {
        self.configurationSHA256 = configurationSHA256; self.indexSHA256 = indexSHA256; self.files = files
    }
}

public struct MiMoV26FilesystemShard: Codable, Sendable {
    public let name: String
    public let objectState: MiMoV26FilesystemObjectState
    public let headerSHA256: String
    public let headerBytes, tensorBytes: Int
    public let tensorKeys: Set<String>
    public let tensorKeysInFileOrder: [String]
    public let tensorLocations: [MiMoV26FilesystemTensorLocation]
    public let suppliedPayloadSHA256: String?
}
public struct MiMoV26FilesystemTensorLocation: Codable, Sendable {
    public let name: String
    /// Absolute byte offset from the beginning of the safetensors file.
    public let dataOffset, byteCount: Int
}
public struct MiMoV26FilesystemLoadPlan: Sendable {
    public let canonicalRoot, configurationURL, indexURL: URL
    public let configurationObject, indexObject: MiMoV26FilesystemObjectState
    public let bundlePlan: MiMoV26ConvertedLoadPlan
    public let shards: [MiMoV26FilesystemShard]
    public let totalFileBytes, metadataBytesRead: Int
    public let expectations: MiMoV26FilesystemExpectations
    /// Known local names only; caller must independently bound/validate reads.
    public var tokenizerURL: URL { canonicalRoot.appendingPathComponent("tokenizer.json") }
    public var tokenizerConfigurationURL: URL { canonicalRoot.appendingPathComponent("tokenizer_config.json") }
}

public enum MiMoV26FilesystemPhase: String, Codable, Sendable {
    case configurationRead, indexRead, headerRead, preflightComplete
    case willCreateShardHandles, shardHandlesCreated, willConstructBundle, bundleConstructed
}
public struct MiMoV26FilesystemProgress: Codable, Sendable {
    public let phase: MiMoV26FilesystemPhase
    public let currentFile: String?
    public let completedShards, totalShards: Int
    /// Actual bytes read by this preflight, excluding native reader internals.
    public let metadataBytesRead: Int
    /// Declared bytes represented by returned handles, NOT materialized bytes.
    public let returnedHandleFileBytes, returnedHandleTensorBytes: Int
    public let totalDeclaredFileBytes, totalDeclaredTensorBytes: Int
}
public struct MiMoV26FilesystemLoadReceipt: Encodable, Sendable {
    public let canonicalRoot: URL
    public let configSHA256, indexSHA256, descriptorSHA256: String
    public let shards: [MiMoV26FilesystemShard]
    public let metadataBytesRead, returnedHandleFileBytes, returnedHandleTensorBytes: Int
    public let payloadDisposition = "lazy native Load handles; no eval, residency or fresh payload-digest proof"
    public let ownershipRequirement = "caller retains immutable source files through materialization and retirement"
}
public struct MiMoV26FilesystemLoadResult {
    public let bundle: MiMoV26ConvertedBundle
    public let receipt: MiMoV26FilesystemLoadReceipt
    fileprivate var materializationHandles: [MiMoV26FilesystemTensorHandle]
    public var materializationHandleCount: Int { materializationHandles.count }
    /// Transfers extra immutable source-handle references in shard/physical
    /// offset order. No bytes are copied or evaluated. The caller drops this
    /// array after explicit materialization; bundle parameters/views continue
    /// owning their dependencies. These source names are not internal module
    /// paths. Treat the arrays as read-only and do not copy this owner to retain
    /// another source-handle collection accidentally.
    public mutating func takeMaterializationHandles() -> [MiMoV26FilesystemTensorHandle] {
        let handles=materializationHandles;materializationHandles=[];return handles
    }
}
public struct MiMoV26FilesystemTensorHandle {
    public let name, shard: String
    public let dataOffset, byteCount: Int
    public let array: MLXArray
}

private func fsHash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
private func fsDigest(_ value: String?) -> Bool {
    value.map { $0.utf8.count == 64 && $0.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) } } ?? true
}
private func fsName(_ name: String) -> Bool {
    !name.isEmpty && name != "." && name != ".." && !name.contains("/") && !name.contains("\\") && !name.utf8.contains(0)
}
private func fsState(_ s: stat, _ name: String) throws -> MiMoV26FilesystemObjectState {
    guard mode_t(s.st_mode) & mode_t(S_IFMT) == mode_t(S_IFREG), let bytes = Int(exactly: s.st_size), bytes >= 0 else {
        throw MiMoV26FilesystemError.unsafeFile(name)
    }
    #if canImport(Darwin)
    let m = s.st_mtimespec, c = s.st_ctimespec
    #else
    let m = s.st_mtim, c = s.st_ctim
    #endif
    return .init(device: String(s.st_dev), inode: String(s.st_ino), bytes: bytes,
                 modifiedSeconds: Int64(m.tv_sec), modifiedNanoseconds: Int64(m.tv_nsec),
                 changedSeconds: Int64(c.tv_sec), changedNanoseconds: Int64(c.tv_nsec))
}
private func fsCurrent(_ url: URL) throws -> MiMoV26FilesystemObjectState {
    var s = stat()
    guard url.isFileURL, lstat(url.path, &s) == 0 else { throw MiMoV26FilesystemError.unsafeFile(url.lastPathComponent) }
    return try fsState(s, url.lastPathComponent)
}
private func fsSum(_ a: Int, _ b: Int, _ label: String) throws -> Int {
    let result = a.addingReportingOverflow(b)
    guard a >= 0, b >= 0, !result.overflow else { throw MiMoV26FilesystemError.limit(label) }
    return result.partialValue
}

/// Existing JSON decoders validate syntax but collapse duplicate object keys.
/// Reject those at every depth before decoding a config/index/header contract.
private func fsUniqueKeys(_ data: Data, _ name: String) throws {
    let bytes = [UInt8](data)
    var stack: [Set<String>?] = [], i = 0
    while i < bytes.count {
        let byte = bytes[i]
        if byte == 123 || byte == 91 {
            stack.append(byte == 123 ? Set<String>() : nil)
            guard stack.count <= 128 else { throw MiMoV26FilesystemError.invalidMetadata("JSON nesting: " + name) }
        } else if byte == 125 || byte == 93 {
            guard !stack.isEmpty else { throw MiMoV26FilesystemError.invalidMetadata(name) }
            stack.removeLast()
        } else if byte == 34 {
            let start = i; i += 1
            while i < bytes.count && bytes[i] != 34 {
                if bytes[i] == 92 { i += 1 }
                i += 1
            }
            guard i < bytes.count else { throw MiMoV26FilesystemError.invalidMetadata(name) }
            var next = i + 1
            while next < bytes.count && [9,10,13,32].contains(bytes[next]) { next += 1 }
            if next < bytes.count && bytes[next] == 58 {
                guard let top = stack.indices.last, stack[top] != nil else { throw MiMoV26FilesystemError.invalidMetadata(name) }
                let key = try JSONDecoder().decode(String.self, from:Data(bytes[start...i]))
                guard stack[top]!.insert(key).inserted else { throw MiMoV26FilesystemError.invalidMetadata("duplicate JSON key: " + name) }
            }
        }
        i += 1
    }
    guard stack.isEmpty else { throw MiMoV26FilesystemError.invalidMetadata(name) }
}

/// Opens only a regular non-symlink leaf. Metadata reads are bounded BEFORE IO.
private final class MiMoV26MetadataFile {
    let url: URL
    let handle: FileHandle
    let state: MiMoV26FilesystemObjectState
    init(_ url: URL) throws {
        self.url = url
        let before = try fsCurrent(url)
        let descriptor = open(url.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw MiMoV26FilesystemError.unsafeFile(url.lastPathComponent) }
        var s = stat()
        guard fstat(descriptor, &s) == 0 else { close(descriptor); throw MiMoV26FilesystemError.unsafeFile(url.lastPathComponent) }
        do {
            let actual = try fsState(s, url.lastPathComponent)
            guard actual == before else { throw MiMoV26FilesystemError.changedObject(url.lastPathComponent) }
            state = actual; handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        } catch { close(descriptor); throw error }
    }
    func validate() throws {
        var s = stat()
        guard fstat(handle.fileDescriptor, &s) == 0,
              try fsState(s, url.lastPathComponent) == state, try fsCurrent(url) == state else {
            throw MiMoV26FilesystemError.changedObject(url.lastPathComponent)
        }
    }
    func read(_ count: Int, cancelled: () -> Bool) throws -> Data {
        guard count >= 0, count <= state.bytes else { throw MiMoV26FilesystemError.invalidMetadata(url.lastPathComponent) }
        var data = Data(); data.reserveCapacity(count)
        while data.count < count {
            if cancelled() { throw MiMoV26FilesystemError.cancelled }
            try validate()
            guard let chunk = try handle.read(upToCount: min(65536,count-data.count)), !chunk.isEmpty else {
                throw MiMoV26FilesystemError.invalidMetadata("short read: " + url.lastPathComponent)
            }
            data.append(chunk)
        }
        try validate(); return data
    }
}

private struct MiMoV26FilesystemIndex: Decodable {
    let weightMap: [String:String]
    enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
}
private struct MiMoV26FilesystemTensor: Decodable {
    let dtype: MiMoV26ConvertedScalarType
    let shape, offsets: [Int]
    enum CodingKeys: String, CodingKey { case dtype, shape, offsets = "data_offsets" }
}
private struct MiMoV26FilesystemHeader: Decodable {
    struct Key: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }
    let tensors: [String:MiMoV26FilesystemTensor]
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy:Key.self)
        var result: [String:MiMoV26FilesystemTensor] = [:]
        for key in c.allKeys {
            if key.stringValue == "__metadata__" { _ = try c.decode([String:String].self,forKey:key) }
            else { result[key.stringValue] = try c.decode(MiMoV26FilesystemTensor.self,forKey:key) }
        }
        tensors = result
    }
}

public enum MiMoV26FilesystemWeights {
    public static func preflight(root: URL, provenance: MiMoV26ConvertedProvenance,
                                 limits: MiMoV26FilesystemLimits, expectations: MiMoV26FilesystemExpectations = .init(),
                                 isCancelled: () -> Bool = { false },
                                 progress: (MiMoV26FilesystemProgress) throws -> Void = { _ in }) throws -> MiMoV26FilesystemLoadPlan {
        guard root.isFileURL else { throw MiMoV26FilesystemError.invalidRoot }
        let canonical = root.standardizedFileURL.resolvingSymlinksInPath()
        var directory = stat()
        guard lstat(canonical.path,&directory) == 0,
              mode_t(directory.st_mode) & mode_t(S_IFMT) == mode_t(S_IFDIR) else { throw MiMoV26FilesystemError.invalidRoot }
        guard [limits.maximumShardBytes,limits.maximumTotalFileBytes,limits.maximumConfigurationBytes,limits.maximumIndexBytes,
               limits.maximumHeaderBytes,limits.maximumMetadataBytes,limits.maximumShards,limits.maximumTensors].allSatisfy({$0>0}),
              limits.maximumConfigurationBytes <= 4_194_304, limits.maximumIndexBytes <= 16_777_216,
              limits.maximumHeaderBytes <= 8_388_608, limits.maximumMetadataBytes <= 67_108_864,
              limits.maximumShards <= 256, limits.maximumTensors <= 16_384,
              fsDigest(expectations.configurationSHA256),fsDigest(expectations.indexSHA256) else {
            throw MiMoV26FilesystemError.limit("metadata limits or expected digest")
        }
        func check() throws { if isCancelled() { throw MiMoV26FilesystemError.cancelled } }
        var metadataBytes = 0, totalFiles = 0, declaredTensorBytes = 0
        func emit(_ phase: MiMoV26FilesystemPhase, _ file: String?, _ done: Int, _ total: Int) throws {
            try check(); try progress(.init(phase:phase,currentFile:file,completedShards:done,totalShards:total,
                metadataBytesRead:metadataBytes,returnedHandleFileBytes:0,returnedHandleTensorBytes:0,
                totalDeclaredFileBytes:phase == .preflightComplete ? totalFiles : 0,
                totalDeclaredTensorBytes:declaredTensorBytes)); try check()
        }
        func readMetadata(_ name: String, maximum: Int) throws -> (Data,MiMoV26FilesystemObjectState) {
            try check(); let file = try MiMoV26MetadataFile(canonical.appendingPathComponent(name))
            guard file.state.bytes <= maximum else { throw MiMoV26FilesystemError.limit(name) }
            let sum = try fsSum(metadataBytes,file.state.bytes,"metadata budget")
            guard sum <= limits.maximumMetadataBytes else { throw MiMoV26FilesystemError.limit("metadata budget") }
            let data = try file.read(file.state.bytes,cancelled:isCancelled); metadataBytes=sum
            try fsUniqueKeys(data,name); return(data,file.state)
        }
        let configURL=canonical.appendingPathComponent("config.json"),indexURL=canonical.appendingPathComponent("model.safetensors.index.json")
        let(config,configState)=try readMetadata("config.json",maximum:limits.maximumConfigurationBytes)
        guard expectations.configurationSHA256.map({$0==fsHash(config)}) ?? true else { throw MiMoV26FilesystemError.invalidMetadata("configuration digest") }
        try emit(.configurationRead,"config.json",0,0)
        let(index,indexState)=try readMetadata("model.safetensors.index.json",maximum:limits.maximumIndexBytes)
        guard expectations.indexSHA256.map({$0==fsHash(index)}) ?? true else { throw MiMoV26FilesystemError.invalidMetadata("index digest") }
        let parsed=try JSONDecoder().decode(MiMoV26FilesystemIndex.self,from:index)
        let names=Set(parsed.weightMap.values)
        guard !parsed.weightMap.isEmpty,parsed.weightMap.count<=limits.maximumTensors,
              !names.isEmpty,names.count<=limits.maximumShards,
              names.allSatisfy({fsName($0)&&$0.hasSuffix(".safetensors")}),Set(expectations.files.keys).isSubset(of:names) else {
            throw MiMoV26FilesystemError.invalidMetadata("root index/file inventory")
        }
        for expected in expectations.files.values {
            guard fsDigest(expected.headerSHA256),fsDigest(expected.payloadSHA256) else { throw MiMoV26FilesystemError.invalidMetadata("expected shard digest") }
        }
        try rootWeights(canonical,expected:names); try emit(.indexRead,"model.safetensors.index.json",0,names.count)
        var descriptors:[String:MiMoV26ConvertedTensorDescriptor]=[:],shards:[MiMoV26FilesystemShard]=[]
        for name in names.sorted() {
            try check(); let file=try MiMoV26MetadataFile(canonical.appendingPathComponent(name))
            guard file.state.bytes>=8,file.state.bytes<=limits.maximumShardBytes else { throw MiMoV26FilesystemError.limit(name) }
            totalFiles=try fsSum(totalFiles,file.state.bytes,"total declared file bytes")
            guard totalFiles<=limits.maximumTotalFileBytes else { throw MiMoV26FilesystemError.limit("total declared file bytes") }
            guard try fsSum(metadataBytes,8,"header prefix")<=limits.maximumMetadataBytes else {
                throw MiMoV26FilesystemError.limit("metadata budget")
            }
            let prefix=try file.read(8,cancelled:isCancelled)
            let rawLength=prefix.enumerated().reduce(UInt64(0)) { $0 | (UInt64($1.element) << ($1.offset*8)) }
            guard rawLength>0,rawLength<=UInt64(limits.maximumHeaderBytes),let length=Int(exactly:rawLength),length<=file.state.bytes-8 else {
                throw MiMoV26FilesystemError.invalidHeader(name)
            }
            let nextBytes=try fsSum(metadataBytes,try fsSum(8,length,"header length"),"metadata budget")
            guard nextBytes<=limits.maximumMetadataBytes else { throw MiMoV26FilesystemError.limit("metadata budget") }
            let raw=try file.read(length,cancelled:isCancelled);metadataBytes=nextBytes
            let digest=fsHash(raw),expected=expectations.files[name]
            guard expected?.objectState.map({$0==file.state}) ?? true,
                  expected?.headerSHA256.map({$0==digest}) ?? true else { throw MiMoV26FilesystemError.changedObject(name) }
            try fsUniqueKeys(raw,name)
            let header=try JSONDecoder().decode(MiMoV26FilesystemHeader.self,from:raw)
            guard !header.tensors.isEmpty,header.tensors.count<=limits.maximumTensors else { throw MiMoV26FilesystemError.invalidHeader(name) }
            var spans:[(Int,Int)]=[],keys=Set<String>(),sum=0
            for(key,tensor)in header.tensors {
                guard descriptors[key]==nil,parsed.weightMap[key]==name,!tensor.shape.isEmpty,
                      tensor.shape.allSatisfy({$0>0&&$0<=Int(Int32.max)}),tensor.offsets.count==2,
                      tensor.offsets[0]>=0,tensor.offsets[1]>=tensor.offsets[0] else { throw MiMoV26FilesystemError.invalidHeader(key) }
                var bytes=tensor.dtype.bytes
                for dimension in tensor.shape { let value=bytes.multipliedReportingOverflow(by:dimension);guard !value.overflow else{throw MiMoV26FilesystemError.invalidHeader(key)};bytes=value.partialValue }
                guard tensor.offsets[1]-tensor.offsets[0]==bytes,tensor.offsets[1]<=file.state.bytes-8-length else { throw MiMoV26FilesystemError.invalidHeader(key) }
                spans.append((tensor.offsets[0],tensor.offsets[1]));sum=try fsSum(sum,bytes,"tensor bytes");keys.insert(key)
                descriptors[key] = .init(shape:tensor.shape,dtype:tensor.dtype,file:name)
            }
            guard descriptors.count<=limits.maximumTensors else { throw MiMoV26FilesystemError.limit("tensor count") }
            var end=0
            for span in spans.sorted(by:{$0.0<$1.0}) { guard span.0==end else{throw MiMoV26FilesystemError.invalidHeader("overlap/gap: "+name)};end=span.1 }
            guard end==file.state.bytes-8-length,keys==Set(parsed.weightMap.filter{$0.value==name}.keys) else { throw MiMoV26FilesystemError.invalidHeader("file closure: "+name) }
            try file.validate()
            let ordered=header.tensors.sorted{$0.value.offsets[0]<$1.value.offsets[0]}
            let orderedKeys=ordered.map(\.key)
            let locations=ordered.map { MiMoV26FilesystemTensorLocation(name:$0.key,
                dataOffset:8+length+$0.value.offsets[0],byteCount:$0.value.offsets[1]-$0.value.offsets[0]) }
            shards.append(.init(name:name,objectState:file.state,headerSHA256:digest,headerBytes:length,tensorBytes:sum,
                tensorKeys:keys,tensorKeysInFileOrder:orderedKeys,tensorLocations:locations,suppliedPayloadSHA256:expected?.payloadSHA256))
            try emit(.headerRead,name,shards.count,names.count)
        }
        let bundlePlan=try MiMoV26ConvertedLoadPlan.make(configurationData:config,indexData:index,descriptors:descriptors,provenance:provenance)
        declaredTensorBytes=bundlePlan.tensorBytes
        let plan=MiMoV26FilesystemLoadPlan(canonicalRoot:canonical,configurationURL:configURL,indexURL:indexURL,
            configurationObject:configState,indexObject:indexState,bundlePlan:bundlePlan,shards:shards,totalFileBytes:totalFiles,
            metadataBytesRead:metadataBytes,expectations:expectations)
        try emit(.preflightComplete,nil,shards.count,names.count)
        try validateCurrentObjects(plan:plan,isCancelled:isCancelled)
        return plan
    }

    /// Recheck at caller materialization boundaries too: creating lazy handles
    /// does not read or freeze payload bytes for later evaluation.
    public static func validateCurrentObjects(plan: MiMoV26FilesystemLoadPlan,
                                               isCancelled: () -> Bool = {false}) throws {
        if isCancelled(){throw MiMoV26FilesystemError.cancelled}
        guard plan.canonicalRoot.standardizedFileURL.resolvingSymlinksInPath()==plan.canonicalRoot,
              try fsCurrent(plan.configurationURL)==plan.configurationObject,try fsCurrent(plan.indexURL)==plan.indexObject else {
            throw MiMoV26FilesystemError.changedObject("root metadata")
        }
        try rootWeights(plan.canonicalRoot,expected:plan.bundlePlan.rootFiles)
        for shard in plan.shards {
            if isCancelled(){throw MiMoV26FilesystemError.cancelled}
            guard try fsCurrent(plan.canonicalRoot.appendingPathComponent(shard.name))==shard.objectState else {
                throw MiMoV26FilesystemError.changedObject(shard.name)
            }
        }
    }

    public static func load(plan: MiMoV26FilesystemLoadPlan, retaining work: NativeConstructionScope,
                            isCancelled: () -> Bool = {false},
                            progress: (MiMoV26FilesystemProgress) throws -> Void = {_ in}) throws -> MiMoV26FilesystemLoadResult {
        try work.withPhase(.filesystemLoad) {
            try loadInScope(plan: plan, work: work, isCancelled: isCancelled, progress: progress)
        }
    }

    private static func loadInScope(plan: MiMoV26FilesystemLoadPlan, work: NativeConstructionScope,
                                    isCancelled: () -> Bool,
                                    progress: (MiMoV26FilesystemProgress) throws -> Void) throws -> MiMoV26FilesystemLoadResult {
        try validateCurrentObjects(plan:plan,isCancelled:isCancelled)
        var tensors:[String:MLXArray]=[:],fileBytes=0,tensorBytes=0,completed=0
        func emit(_ phase:MiMoV26FilesystemPhase,_ name:String?) throws {
            if isCancelled(){throw MiMoV26FilesystemError.cancelled}
            try progress(.init(phase:phase,currentFile:name,completedShards:completed,totalShards:plan.shards.count,
                metadataBytesRead:plan.metadataBytesRead,returnedHandleFileBytes:fileBytes,returnedHandleTensorBytes:tensorBytes,
                totalDeclaredFileBytes:plan.totalFileBytes,totalDeclaredTensorBytes:plan.bundlePlan.tensorBytes))
            if isCancelled(){throw MiMoV26FilesystemError.cancelled}
        }
        for shard in plan.shards {
            try emit(.willCreateShardHandles,shard.name)
            let url=plan.canonicalRoot.appendingPathComponent(shard.name)
            guard try fsCurrent(url)==shard.objectState else { throw MiMoV26FilesystemError.changedObject(shard.name) }
            let values:[String:MLXArray]
            // Pinned safetensors reader builds lazy Load nodes, not payload work.
            // Capture the exact CPU stream and returned roots before any veto.
            let sourceStream = StreamOrDevice.cpu
            try work.capture(sourceStream.stream)
            do { values=try loadArrays(url:url,stream:sourceStream) }
            catch { throw MiMoV26FilesystemError.nativeLoad(shard.name) }
            try work.retain(arrays: values.values)
            try work.checkpoint("filesystem.returnedSourceHandles")
            guard try fsCurrent(url)==shard.objectState else { throw MiMoV26FilesystemError.changedObject(shard.name) }
            guard Set(values.keys)==shard.tensorKeys else { throw MiMoV26FilesystemError.invalidHeader("native key closure: "+shard.name) }
            for(key,value)in values {
                guard let d=plan.bundlePlan.descriptors[key],value.shape==d.shape,value.dtype==d.dtype.dtype,tensors[key]==nil else {
                    throw MiMoV26FilesystemError.invalidHeader("native handle: "+key)
                }
                tensors[key]=value
            }
            completed+=1;fileBytes=try fsSum(fileBytes,shard.objectState.bytes,"returned handles");tensorBytes=try fsSum(tensorBytes,shard.tensorBytes,"returned tensor handles")
            try emit(.shardHandlesCreated,shard.name)
        }
        try emit(.willConstructBundle,nil);try validateCurrentObjects(plan:plan,isCancelled:isCancelled)
        try work.capture(StreamOrDevice.default.stream)
        let bundle=try MiMoV26ConvertedWeights.load(plan:plan.bundlePlan,tensors:tensors)
        try work.retainOwner(bundle.target); try work.retainOwner(bundle.vision)
        try work.retainOwner(bundle.audioPatch); try work.retainOwner(bundle.mtp)
        try work.checkpoint("filesystem.constructedBundle")
        try emit(.bundleConstructed,nil);try validateCurrentObjects(plan:plan,isCancelled:isCancelled)
        let handles=plan.shards.flatMap { shard in shard.tensorLocations.map { location in
            MiMoV26FilesystemTensorHandle(name:location.name,shard:shard.name,dataOffset:location.dataOffset,
                byteCount:location.byteCount,array:tensors[location.name]!)
        } }
        return .init(bundle:bundle,receipt:.init(canonicalRoot:plan.canonicalRoot,configSHA256:plan.bundlePlan.configSHA256,
            indexSHA256:plan.bundlePlan.indexSHA256,descriptorSHA256:plan.bundlePlan.descriptorSHA256,shards:plan.shards,
            metadataBytesRead:plan.metadataBytesRead,returnedHandleFileBytes:fileBytes,returnedHandleTensorBytes:tensorBytes),
            materializationHandles:handles)
    }

    private static func rootWeights(_ root:URL,expected:Set<String>) throws {
        guard let entries=FileManager.default.enumerator(at:root,includingPropertiesForKeys:nil,options:[.skipsSubdirectoryDescendants]) else {
            throw MiMoV26FilesystemError.invalidRoot
        }
        // Other tokenizer/processor/provenance files and separate subdirectories
        // are outside the root weight-map scope. Any unindexed root payload
        // format is rejected, not silently skipped by extension filtering.
        var count=0,payloads=Set<String>()
        while let url=entries.nextObject() as? URL {
            count+=1
            guard count<=4096 else{throw MiMoV26FilesystemError.limit("root directory entries")}
            if ["safetensors","npz","bin","pt","pth"].contains(url.pathExtension.lowercased()) { payloads.insert(url.lastPathComponent) }
        }
        guard payloads==expected else { throw MiMoV26FilesystemError.invalidMetadata("unindexed/missing root payload files") }
    }
}
