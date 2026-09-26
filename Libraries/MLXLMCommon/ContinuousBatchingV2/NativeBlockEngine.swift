import Foundation

/// Native block execution behind the existing provider engine protocol.
/// Tensor/session state is confined to `queue`; cross-thread controls and
/// capacity counters are protected by `lock`. Requests are stepped round-robin.
/// This is concurrent admission, NOT a claim of rectangular GPU batch fusion.
public final class CBv2NativeBlockEngine: CBv2Engine, @unchecked Sendable {
    // Stable across retained metadata; unlike an address, never recycled when
    // an old engine dies before a transfer owner is rebound to a new engine.
    let checkpointOwnerIdentity = UUID()
    public typealias SessionFactory =
        @Sendable (
            CBv2Request, CBv2NativeBlockCancellation
        ) throws -> any CBv2NativeBlockSession

    private final class Control: @unchecked Sendable {
        let request: CBv2Request
        let generation = UUID()
        let cancellation = CBv2NativeBlockCancellation()
        let reservation: Int
        let submitted = DispatchTime.now().uptimeNanoseconds
        var output: CBv2OutputStream!
        // Engine lock protects these fields.
        var running = false
        var paused = false
        var retainedBytes = 0
        var sharedStorageBytes = 0
        var activeTokens = 0
        var watchdogReason: CBv2FinishReason?
        var usageSnapshot: CBv2Usage
        // Engine queue owns lease transitions; the timer never touches these.
        var lease: CBv2RequestLeaseState
        var completedWork = 0
        init(request: CBv2Request, reservation: Int, config: CBv2EngineLoopConfig) {
            self.request = request
            self.reservation = reservation
            usageSnapshot = .init(promptTokens: request.promptTokens.count, completionTokens: 0)
            let now = config.clock.now()
            lease = config.useLegacyRequestTimeout ? .legacy(now: now, wall: config.requestTimeout)
                : .init(now: now, admissionLease: config.admissionLease,
                    prefillLease: config.prefillProgressLease, decodeLease: config.decodeProgressLease,
                    backpressureLease: config.backpressureLease,
                    safety: CBv2SafetyCeiling.duration(promptTokens: request.promptTokens.count,
                        maxTokens: request.maxTokens, admissionLease: config.admissionLease,
                        decodeFloorTPS: config.safetyCeilingDecodeFloorTPS),
                    computedTokens: 0, generatedTokens: 0)
        }
    }
    private final class Row {
        let control: Control
        let session: any CBv2NativeBlockSession
        let decoder: CBv2NativeBlockTextDecoder
        var timing = CBv2RequestTiming()
        var committedTokens = 0
        // With caller stop strings, text may be stable before token ownership
        // is: UTF-8/cleanup can hold a suffix across blocks. Defer raw IDs until
        // terminal reconciliation so a later match never retracts prior IDs.
        var pendingStopTokens = [Int]()
        init(control: Control, session: any CBv2NativeBlockSession, tokenizer: any Tokenizer) {
            self.control = control
            self.session = session
            decoder = CBv2NativeBlockTextDecoder(
                tokenizer: tokenizer, stopStrings: control.request.stopStrings)
            timing.admittedNanos = max(1, DispatchTime.now().uptimeNanoseconds &- control.submitted)
        }
    }

    private let lock = NSLock()
    private let queue = DispatchQueue(label: "mlx.native-block-engine", qos: .userInitiated)
    private let tokenizer: any Tokenizer
    private let reservationForRequest: @Sendable (CBv2Request) throws -> Int
    private let maxConcurrentRequests: Int
    private let maxWaiting: Int
    private let outputBufferCapacity: Int
    private let shutdownGraceSeconds: Double
    private let loopConfig: CBv2EngineLoopConfig
    private let watchdogQueue = DispatchQueue(label: "mlx.native-block-watchdog", qos: .utility)
    private var watchdogTimer: DispatchSourceTimer?
    // Engine lock protects watchdog state, but never GPU/session arrays.
    private var stepStartedAt: ContinuousClock.Instant?
    private var watchdogWakeQueued = false
    private var healthy = true
    private var wedgeCallback: (@Sendable (TimeInterval) -> Void)?
    public var isHealthy: Bool { lock.withLock { healthy } }
    public var onStepWedge: (@Sendable (TimeInterval) -> Void)? {
        get { lock.withLock { wedgeCallback } }
        set { lock.withLock { wedgeCallback = newValue } }
    }
    private let sharedResources: CBv2NativeBlockSharedResources?
    package let pagedMemory: CBv2NativeBlockPagedMemory?
    private var pagedBackingBytes = 0
    public var usesPagedStorage: Bool { pagedMemory != nil }
    public var usesProcessMemoryOwner: Bool { pagedMemory?.usesProcessMemoryOwner == true }
    public let completeNativePrefixCache: (any CBv2NativeBlockPrefixCache)?
    private let checkpointPlanner: CBv2NativeBlockCheckpointPlanner?
    private var sharedRetainedBytes = 0
    private var sharedReservationBytes = 0
    private var checkpointReservationBytes = 0
    private var accepting = true
    private var bytesCapacity: Int
    private var controls = [CBv2RequestID: Control]()
    private var stepsExecuted = 0
    private var stepWallNanos: UInt64 = 0
    private var decodedRows: UInt64 = 0
    // Queue-confined state, including the only factory/weight owner.
    private var factory: SessionFactory?
    private var waiting = [Control]()
    private var rows = [Row]()
    private var cursor = 0
    private var pumpScheduled = false
    private var draining = false
    private var shutdownWaiters = [CheckedContinuation<Void, Never>]()

    public init(
        tokenizer: any Tokenizer, kvBytesCapacity: Int,
        maxConcurrentRequests: Int = 4, maxWaiting: Int = 64, outputBufferCapacity: Int = 8,
        shutdownGraceSeconds: Double = 10,
        loopConfig: CBv2EngineLoopConfig = .init(),
        sharedResources: CBv2NativeBlockSharedResources? = nil,
        pagedMemory: CBv2NativeBlockPagedMemory? = nil,
        completeNativePrefixCache: (any CBv2NativeBlockPrefixCache)? = nil,
        checkpointPlanner: CBv2NativeBlockCheckpointPlanner? = nil,
        reservationForRequest: @escaping @Sendable (CBv2Request) throws -> Int,
        makeSession: @escaping SessionFactory
    ) throws {
        guard kvBytesCapacity > 0, maxConcurrentRequests > 0, maxWaiting >= 0,
            !maxConcurrentRequests.addingReportingOverflow(maxWaiting).overflow,
            outputBufferCapacity > 0, shutdownGraceSeconds.isFinite,
            shutdownGraceSeconds >= 0, shutdownGraceSeconds <= 3600
        else { throw CBv2NativeBlockError.invalidConfiguration }
        let intervals = [loopConfig.requestTimeout, loopConfig.admissionLease,
            loopConfig.prefillProgressLease, loopConfig.decodeProgressLease,
            loopConfig.backpressureLease, loopConfig.stepTimeout]
        guard intervals.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1e9 }),
            loopConfig.watchdogInterval.isFinite, loopConfig.watchdogInterval > 0,
            loopConfig.watchdogInterval <= 60,
            loopConfig.safetyCeilingDecodeFloorTPS.isFinite,
            loopConfig.safetyCeilingDecodeFloorTPS > 0 else {
            throw CBv2NativeBlockError.invalidConfiguration
        }
        guard
            sharedResources == nil
                || (sharedResources!.maximumBytes > 0
                    && sharedResources!.maximumBytes < kvBytesCapacity)
        else {
            throw CBv2NativeBlockError.invalidConfiguration
        }
        self.tokenizer = tokenizer
        self.bytesCapacity = kvBytesCapacity
        self.maxConcurrentRequests = maxConcurrentRequests
        self.maxWaiting = maxWaiting
        self.outputBufferCapacity = outputBufferCapacity
        self.shutdownGraceSeconds = shutdownGraceSeconds
        self.loopConfig = loopConfig
        self.sharedResources = sharedResources
        try pagedMemory?.claimEngine(capacity: kvBytesCapacity)
        self.pagedMemory = pagedMemory
        guard (completeNativePrefixCache == nil) == (checkpointPlanner == nil) else {
            throw CBv2NativeBlockError.invalidConfiguration
        }
        self.completeNativePrefixCache = completeNativePrefixCache
        self.checkpointPlanner = checkpointPlanner
        self.sharedReservationBytes = sharedResources?.maximumBytes ?? 0
        if let pagedMemory, let sharedResources {
            do { try pagedMemory.setSharedReservation(bytes: sharedResources.maximumBytes) } catch {
                self.sharedReservationBytes = 0
            }
        }
        self.reservationForRequest = reservationForRequest
        self.factory = makeSession
        startWatchdog()
    }

    public func submit(_ request: CBv2Request) throws -> AsyncStream<CBv2Event> {
        try submitWithRetirement(request).events
    }

    /// A terminal stream is not proof that a blocked native step released its
    /// arrays. External memory/cache owners retain this exact generation's loan.
    public func submitWithRetirement(_ request: CBv2Request) throws -> (
        events: AsyncStream<CBv2Event>, retirement: CBv2RequestRetirement
    ) {
        guard !request.promptTokens.isEmpty, request.maxTokens > 0,
            request.stopStrings.allSatisfy({ !$0.isEmpty })
        else {
            throw CBv2NativeBlockError.unsupportedRequest("prompt/output/stop bounds")
        }
        // Model-specific validation and a conservative full-request reservation
        // happen BEFORE enqueue, including unimplemented sampler/media controls.
        var request = request
        let cacheCanBeFunded = lock.withLock {
            sharedResources.map {
                $0.maximumBytes < bytesCapacity
                    && (pagedMemory == nil || sharedReservationBytes >= $0.maximumBytes)
            } ?? false
        }
        var coldRequest = request
        coldRequest.prefixCacheEnabled = false
        let coldBytes = try reservationForRequest(coldRequest)
        let desired = cacheCanBeFunded ? try reservationForRequest(request) : coldBytes
        let claimed = request.nativeReservationBytes ?? desired
        guard claimed >= coldBytes else {
            throw CBv2KVError.capacityExhausted(needed: coldBytes, available: max(0, claimed))
        }
        if !cacheCanBeFunded || claimed < desired { request.prefixCacheEnabled = false }
        let bytes = try reservationForRequest(request)
        guard bytes > 0 else { throw CBv2NativeBlockError.invalidConfiguration }
        let control = Control(request: request, reservation: bytes, config: loopConfig)
        let generation = control.generation
        control.output = CBv2OutputStream(
            id: request.id, capacity: outputBufferCapacity,
            onBackpressure: { [weak self] id, paused in
                self?.setPaused(id, generation: generation, paused: paused)
            },
            onAbandoned: { [weak self] id in
                self?.cancel(id, generation: generation)
            })
        try lock.withLock {
            guard accepting, healthy else { throw CBv2NativeBlockError.shuttingDown }
            guard controls[request.id] == nil else { throw CBv2NativeBlockError.duplicateRequest }
            let reserved = controls.values.reduce(
                sharedReservationBytes + checkpointReservationBytes
            ) { $0 + $1.reservation }
            guard controls.count < maxConcurrentRequests + maxWaiting,
                bytes <= max(0, bytesCapacity - reserved)
            else {
                throw CBv2KVError.capacityExhausted(
                    needed: bytes, available: max(0, bytesCapacity - reserved))
            }
            try pagedMemory?.reserve(
                id: request.id,
                tokens: request.promptTokens.count + request.maxTokens, totalBytes: bytes)
            controls[request.id] = control
        }
        queue.async { [self] in
            if draining { control.cancellation.cancel() }
            waiting.append(control)
            schedulePump()
        }
        return (control.output.makeStream(), .init(stream: control.output))
    }

    /// The provider's process-global ledger uses this SAME request estimate,
    /// avoiding a second architecture formula that undercharges or double-counts.
    public func estimatedRequestBytes(_ request: CBv2Request) throws -> Int {
        var request = request
        if !lock.withLock({
            sharedResources.map {
                $0.maximumBytes < bytesCapacity
                    && (pagedMemory == nil || sharedReservationBytes >= $0.maximumBytes)
            } ?? false
        }) {
            request.prefixCacheEnabled = false
        }
        return try reservationForRequest(request)
    }

    public func cancel(_ id: CBv2RequestID) { cancel(id, generation: nil) }

    /// Independent transfer owners share the SAME slot grant as active
    /// requests and resident cache. This does not create process-global credit;
    /// provider I/O and destinations must also retain their process reservation.
    public func reserveNativeCheckpoint(bytes: Int) throws -> CBv2NativeBlockCheckpointLease {
        var pagedLease: CBv2CheckpointReservation?
        try lock.withLock {
            guard accepting, bytes > 0 else { throw CBv2NativeBlockError.shuttingDown }
            let reserved = controls.values.reduce(
                sharedReservationBytes + checkpointReservationBytes
            ) { $0 + $1.reservation }
            guard bytes <= max(0, bytesCapacity - reserved) else {
                throw CBv2KVError.capacityExhausted(
                    needed: bytes, available: max(0, bytesCapacity - reserved))
            }
            pagedLease = try pagedMemory?.reserveTransient(bytes: bytes)
            checkpointReservationBytes += bytes
        }
        let retainedLease = pagedLease
        return .init(bytes: bytes) { [weak self, retainedLease] in
            retainedLease?.release()
            guard let self else { return }
            self.lock.withLock { self.checkpointReservationBytes -= bytes }
        }
    }

    public func reserveNativeCheckpointReadScratch() throws -> CBv2CompleteCheckpointIOLease {
        if usesProcessMemoryOwner {
            guard lock.withLock({ accepting }) else { throw CBv2NativeBlockError.shuttingDown }
            // Provider-owned host I/O has its own global reservation. Native
            // allocation receives a separate exact charge after authentication.
            return .init(reservation: .init(onRelease: {}), usesProcessMemoryOwner: true)
        }
        let lease = try reserveNativeCheckpoint(
            bytes: CBv2CompleteCheckpointManifest.maximumProviderScratchBytes)
        return .init(reservation: .init { lease.close() })
    }

    func coverNativeCheckpointAllocation(bytes: Int) throws -> CBv2MemoryCoverage? {
        try pagedMemory?.admission.coverEvaluatedAllocation(bytes: bytes)
    }

    public func planNativeCheckpointImport(
        manifest: CBv2CompleteCheckpointManifest,
        request: CBv2Request
    ) throws -> CBv2NativeBlockCheckpointImportPlan {
        guard let checkpointPlanner, lock.withLock({ accepting }) else {
            throw CBv2NativeBlockError.unsupportedRequest("native checkpoint import")
        }
        // Run the same full request validation as ordinary admission before
        // creating any manifest/destination ownership.
        _ = try reservationForRequest(request)
        return try checkpointPlanner(manifest, request, self)
    }
    private func cancel(_ id: CBv2RequestID, generation: UUID?) {
        lock.withLock {
            if let control = controls[id], generation == nil || control.generation == generation {
                control.cancellation.cancel()
            }
        }
        queue.async { [self] in schedulePump() }
    }
    private func setPaused(_ id: CBv2RequestID, generation: UUID, paused: Bool) {
        let now = loopConfig.clock.now()
        lock.withLock {
            guard let control = controls[id], control.generation == generation else { return }
            control.paused = paused
        }
        queue.async { [self] in
            guard let control = lock.withLock({ controls[id] }),
                control.generation == generation,
                lock.withLock({ control.paused == paused }) else { return }
            if paused { control.lease.markPaused(now: now) }
            else { control.lease.markResumed(now: now) }
            schedulePump()
        }
    }
    public func capacity() -> CBv2CapacitySnapshot {
        lock.withLock {
            let active = controls.values.filter(\.running).count
            return CBv2CapacitySnapshot(
                activeRequests: active, waitingRequests: controls.count - active,
                kvBytesInUse: controls.values.reduce(
                    sharedRetainedBytes + checkpointReservationBytes + pagedBackingBytes
                ) {
                    $0 + $1.retainedBytes - $1.sharedStorageBytes
                },
                kvBytesCapacity: bytesCapacity, kvBytesBackendCapacity: bytesCapacity,
                kvBytesReserved: pagedMemory?.reservedBytes
                    ?? controls.values.reduce(sharedReservationBytes + checkpointReservationBytes) {
                        $0 + $1.reservation
                    },
                activeTokens: controls.values.reduce(0) { $0 + $1.activeTokens },
                stepsExecuted: stepsExecuted, stepWallNanosTotal: stepWallNanos,
                decodeRowsTotal: decodedRows)
        }
    }
    var pausedRequestCountForTesting: Int { lock.withLock { controls.values.filter(\.paused).count } }
    public func updateKVBytesCapacity(_ bytes: Int) {
        lock.withLock {
            bytesCapacity = max(0, bytes)
            pagedMemory?.updateCapacity(bytesCapacity)
            if let sharedResources, sharedResources.maximumBytes < bytesCapacity, accepting {
                if pagedMemory == nil {
                    sharedReservationBytes = max(
                        sharedReservationBytes, sharedResources.maximumBytes)
                } else if (try? pagedMemory?.setSharedReservation(
                    bytes: sharedResources.maximumBytes)) != nil
                {
                    sharedReservationBytes = max(
                        sharedReservationBytes, sharedResources.maximumBytes)
                }
            }
            // Do not release an old cache promise until its actual trim.
        }
        queue.async { [self] in reconcileSharedResources() }
    }

    private func reconcileSharedResources() {
        if let pagedMemory {
            let backing = pagedMemory.backend.bytesWired
            lock.withLock { pagedBackingBytes = backing }
        }
        guard let sharedResources else { return }
        if draining && lock.withLock({ controls.isEmpty }) {
            sharedResources.clear()
            try? pagedMemory?.setSharedReservation(bytes: 0)
            lock.withLock {
                sharedRetainedBytes = 0
                sharedReservationBytes = 0
            }
            return
        }
        let capacity = lock.withLock { bytesCapacity }
        var budget = sharedResources.maximumBytes < capacity ? sharedResources.maximumBytes : 0
        if budget > 0, let pagedMemory {
            do { try pagedMemory.setSharedReservation(bytes: budget) } catch { budget = 0 }
        }
        sharedResources.trim(budget)
        let retained = max(0, sharedResources.retainedBytes())
        if budget == 0 && retained == 0 { try? pagedMemory?.setSharedReservation(bytes: 0) }
        lock.withLock {
            sharedRetainedBytes = retained
            let latestBudget =
                sharedResources.maximumBytes < bytesCapacity ? sharedResources.maximumBytes : 0
            if latestBudget == budget || (pagedMemory != nil && budget == 0) {
                sharedReservationBytes = max(budget, retained)
            } else {
                sharedReservationBytes = max(sharedReservationBytes, latestBudget, retained)
                queue.async { [self] in reconcileSharedResources() }
            }
        }
    }

    private func schedulePump() {
        guard !pumpScheduled else { return }
        let runnable =
            !waiting.isEmpty
            || rows.contains { row in
                row.control.cancellation.isCancelled || !lock.withLock { row.control.paused }
            }
        guard runnable else {
            completeDrainIfIdle()
            return
        }
        pumpScheduled = true
        queue.async { [self] in pump() }
    }
    private func pump() {
        pumpScheduled = false
        reconcileSharedResources()
        expireLeases()
        for control in waiting where control.cancellation.isCancelled {
            finishWaiting(control, reason: terminalReason(control) ?? .cancelled)
        }
        waiting.removeAll { $0.cancellation.isCancelled }
        for row in rows where row.control.cancellation.isCancelled {
            finish(row, reason: terminalReason(row.control) ?? .cancelled)
        }
        waiting.sort {
            $0.request.priority == $1.request.priority
                ? $0.submitted < $1.submitted : $0.request.priority > $1.request.priority
        }
        while rows.count < maxConcurrentRequests, !waiting.isEmpty {
            let control = waiting.removeFirst()
            // The independent watchdog may have cancelled the waiting cohort
            // while a preceding session factory was blocked.
            if control.cancellation.isCancelled {
                finishWaiting(control, reason: terminalReason(control) ?? .cancelled)
                continue
            }
            guard let factory else {
                finishWaiting(control, reason: .cancelled)
                continue
            }
            do {
                control.lease.markAdmitted(now: loopConfig.clock.now())
                beginQuantum()
                let row = Row(
                    control: control, session: try factory(control.request, control.cancellation),
                    tokenizer: tokenizer)
                endQuantum()
                rows.append(row)
                lock.withLock { control.running = true }
            } catch {
                endQuantum()
                finishWaiting(control, reason: terminalReason(control) ?? .error("native_block_initialization_failed"))
            }
        }
        expireLeases()
        guard !rows.isEmpty else {
            completeDrainIfIdle()
            return
        }
        var selected: Row?
        for _ in 0 ..< rows.count {
            cursor %= rows.count
            let row = rows[cursor]
            cursor = (cursor + 1) % rows.count
            if !lock.withLock({ row.control.paused }) {
                selected = row
                break
            }
        }
        guard let row = selected else {
            completeDrainIfIdle()
            return
        }
        let before = DispatchTime.now().uptimeNanoseconds
        do {
            if row.control.cancellation.isCancelled {
                finish(row, reason: terminalReason(row.control) ?? .cancelled)
                schedulePump()
                return
            }
            beginQuantum()
            let step = try row.session.advanceNative()
            endQuantum()
            let ended = DispatchTime.now().uptimeNanoseconds
            guard row.session.retainedBytes >= 0,
                row.session.retainedBytes <= row.control.reservation,
                row.session.sharedStorageBytes >= 0,
                row.session.sharedStorageBytes <= row.session.retainedBytes
            else {
                throw CBv2KVError.capacityExhausted(
                    needed: row.session.retainedBytes, available: row.control.reservation)
            }
            let physicalPages = pagedMemory?.backend.bytesWired ?? 0
            lock.withLock {
                stepsExecuted += 1
                pagedBackingBytes = physicalPages
                stepWallNanos &+= ended &- before
                row.control.retainedBytes = row.session.retainedBytes
                row.control.sharedStorageBytes = row.session.sharedStorageBytes
                row.control.activeTokens = row.session.activeTokenCount
            }
            row.timing.stepLatencyNanosSum &+= ended &- before
            row.timing.stepLatencyNanosMax = max(row.timing.stepLatencyNanosMax, ended &- before)
            row.timing.batchRowsSum += 1
            row.timing.batchRowsMin = 1
            row.timing.batchRowsMax = 1
            expireLeases()
            if row.control.cancellation.isCancelled {
                finish(row, reason: terminalReason(row.control) ?? .cancelled)
            } else {
                try consume(step, row: row, before: before, ended: ended)
                // Confirmed work refreshes liveness without turning a draft
                // canvas into a generated token or public stream event.
                row.control.completedWork += 1
                let phase: CBv2RequestLeaseState.Phase
                if case .prefill(_, let complete) = step { phase = complete ? .decode : .prefill }
                else { phase = .decode }
                row.control.lease.recordNativeProgress(now: loopConfig.clock.now(),
                    phase: phase, completedWork: row.control.completedWork)
                var usage = row.session.prefixUsage
                usage.promptTokens = row.control.request.promptTokens.count
                usage.completionTokens = row.committedTokens
                usage.timing = row.timing
                lock.withLock { row.control.usageSnapshot = usage }
            }
        } catch is CancellationError {
            endQuantum()
            finish(row, reason: terminalReason(row.control) ?? .cancelled)
        } catch {
            endQuantum()
            finish(row, reason: terminalReason(row.control) ?? .error("native_block_execution_failed"))
        }
        reconcileSharedResources()
        schedulePump()
    }

    private func consume(_ step: CBv2NativeBlockStep, row: Row, before: UInt64, ended: UInt64)
        throws
    {
        switch step {
        case .prefill(let computed, let complete):
            guard row.session.generatedTokenCount == row.committedTokens else {
                throw CBv2NativeBlockError.unsupportedRequest("provisional token accounting")
            }
            if row.timing.prefillFirstLaunchNanos == 0 {
                row.timing.prefillFirstLaunchNanos = max(1, before &- row.control.submitted)
                row.timing.kvAllocatedNanos = row.timing.prefillFirstLaunchNanos
            }
            if computed > 0 { row.timing.prefillChunks += 1 }
            row.timing.prefillChunkTokensMax = max(
                row.timing.prefillChunkTokensMax, UInt32(clamping: computed))
            if complete { row.timing.promptComputedNanos = max(1, ended &- row.control.submitted) }
        case .progress:
            guard row.session.generatedTokenCount == row.committedTokens else {
                throw CBv2NativeBlockError.unsupportedRequest("provisional token accounting")
            }
        case .committed(let tokens, let stopToken, let finishReason):
            let raw = tokens + (stopToken.map { [$0] } ?? [])
            guard raw.allSatisfy({ $0 >= 0 }),
                raw.count <= row.control.request.maxTokens - row.committedTokens,
                row.session.generatedTokenCount == row.committedTokens + raw.count,
                finishReason == nil || finishReason == .stop || finishReason == .length
            else {
                throw CBv2NativeBlockError.unsupportedRequest("committed block accounting")
            }
            row.committedTokens += raw.count
            let defersTokens = !row.control.request.stopStrings.isEmpty
            if defersTokens { row.pendingStopTokens.append(contentsOf: raw) }
            if !raw.isEmpty {
                if row.timing.firstTokenNanos == 0 {
                    row.timing.firstTokenNanos = max(1, ended &- row.control.submitted)
                } else {
                    row.timing.decodeSteps += 1
                    lock.withLock { decodedRows += 1 }
                }
            }
            let text = try row.decoder.append(tokens, terminal: finishReason != nil)
            var deliveredTokens = defersTokens ? [] : raw
            if row.decoder.matchedStopString {
                guard let count = row.decoder.stopTokenCount,
                    count > 0, count <= row.pendingStopTokens.count else {
                    throw CBv2NativeBlockError.unsupportedRequest("stop output accounting")
                }
                row.committedTokens = count
                deliveredTokens = Array(row.pendingStopTokens.prefix(count))
                row.pendingStopTokens.removeAll()
            } else if finishReason != nil && defersTokens {
                deliveredTokens = row.pendingStopTokens
                row.pendingStopTokens.removeAll()
            }
            if !deliveredTokens.isEmpty || !text.isEmpty {
                row.control.output.emit(.delta(text: text, tokens: deliveredTokens, logprobs: nil))
            }
            if row.decoder.matchedStopString {
                finish(row, reason: .stop)
            } else if let finishReason {
                finish(row, reason: finishReason)
            }
        }
    }
    private func finishWaiting(_ control: Control, reason: CBv2FinishReason) {
        pagedMemory?.release(id: control.request.id)
        retireControl(control)
        control.output.releaseEngineOwnership()
        control.output.finish(
            reason: reason,
            usage: .init(promptTokens: control.request.promptTokens.count, completionTokens: 0))
    }
    private func finish(_ row: Row, reason: CBv2FinishReason) {
        // Cancellation/error can end a request between committed blocks. Flush
        // only previously finalized IDs, never the session's provisional canvas.
        // A matched stop already clipped and consumed this buffer above.
        if !row.pendingStopTokens.isEmpty {
            row.control.output.emit(.delta(text: "", tokens: row.pendingStopTokens, logprobs: nil))
            row.pendingStopTokens.removeAll()
        }
        let completion = row.committedTokens
        var usage = row.session.prefixUsage
        row.session.finish(reason: reason)
        row.session.cancel()
        rows.removeAll { $0 === row }
        pagedMemory?.release(id: row.control.request.id)
        retireControl(row.control)
        row.timing.finishedNanos = max(
            1, DispatchTime.now().uptimeNanoseconds &- row.control.submitted)
        usage.promptTokens = row.control.request.promptTokens.count
        usage.completionTokens = completion
        usage.timing = row.timing
        reconcileSharedResources()
        row.control.output.releaseEngineOwnership()
        row.control.output.finish(reason: reason, usage: usage)
    }

    /// Reject new work and cancel queued requests. Running requests may drain;
    /// after the grace interval cancellation is observed at the next quantum.
    /// Never report release while an in-flight device quantum still owns state.
    public func shutdown() async {
        lock.withLock { accepting = false }
        // Providers normally close first, but direct SDK owners must also
        // cancel pending transfers. Never call user/store code under our lock.
        completeNativePrefixCache?.close()
        await withCheckedContinuation { continuation in
            queue.async { [self] in
                draining = true
                shutdownWaiters.append(continuation)
                waiting.forEach { $0.cancellation.cancel() }
                queue.asyncAfter(deadline: .now() + shutdownGraceSeconds) { [weak self] in
                    guard let self, self.draining else { return }
                    self.rows.forEach { $0.control.cancellation.cancel() }
                    self.schedulePump()
                }
                schedulePump()
                completeDrainIfIdle()
            }
        }
    }
    private func completeDrainIfIdle() {
        guard draining, lock.withLock({ controls.isEmpty }) else { return }
        watchdogTimer?.cancel()
        factory = nil
        sharedResources?.clear()
        try? pagedMemory?.setSharedReservation(bytes: 0)
        lock.withLock {
            sharedRetainedBytes = 0
            sharedReservationBytes = 0
            pagedBackingBytes = pagedMemory?.backend.bytesWired ?? 0
        }
        let waiters = shutdownWaiters
        shutdownWaiters.removeAll()
        waiters.forEach { $0.resume() }
    }

    private func retireControl(_ control: Control) {
        lock.withLock {
            controls.removeValue(forKey: control.request.id)
            if controls.isEmpty, stepStartedAt == nil { healthy = true }
        }
    }

    private func terminalReason(_ control: Control) -> CBv2FinishReason? {
        lock.withLock { control.watchdogReason }
    }

    /// Engine-queue-only lease scan. A blocked GPU quantum is handled by the
    /// independent watchdog; ordinary progress expiry retires at a safe boundary.
    private func expireLeases() {
        let now = loopConfig.clock.now()
        for control in lock.withLock({ Array(controls.values) }) {
            guard !control.cancellation.isCancelled,
                let cause = control.lease.expiredCause(now: now,
                    isRunning: lock.withLock({ control.running }),
                    isPaused: lock.withLock({ control.paused })) else { continue }
            let reason: CBv2FinishReason = cause == .legacyRequestTimeout
                ? .error("request exceeded \(Int(loopConfig.requestTimeout))s deadline")
                : .terminal(cause: cause, message: cause.diagnostic)
            lock.withLock { control.watchdogReason = reason }
            control.cancellation.cancel()
        }
    }

    private func beginQuantum() { lock.withLock { stepStartedAt = loopConfig.clock.now() } }
    private func endQuantum() {
        // A delayed polling callback cannot excuse an already-overdue quantum.
        watchdogTick(scheduleWake: false)
        lock.withLock { stepStartedAt = nil }
    }

    private func startWatchdog() {
        let timer = DispatchSource.makeTimerSource(queue: watchdogQueue)
        timer.schedule(deadline: .now() + loopConfig.watchdogInterval, repeating: loopConfig.watchdogInterval)
        timer.setEventHandler { [weak self] in self?.watchdogTick() }
        timer.resume()
        watchdogTimer = timer
    }

    private func watchdogTick(scheduleWake: Bool = true) {
        var terminals: [(Control, CBv2Usage)] = []
        var callback: (@Sendable (TimeInterval) -> Void)?
        var elapsed = 0.0
        let wake = lock.withLock {
            if let start = stepStartedAt, healthy {
                let duration = loopConfig.clock.now() - start
                elapsed = Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
                if elapsed > loopConfig.stepTimeout {
                    healthy = false
                    callback = wedgeCallback
                    for control in controls.values {
                        control.watchdogReason = .terminal(cause: .watchdog, message: CBv2TerminalCause.watchdog.diagnostic)
                        control.cancellation.cancel()
                        terminals.append((control, control.usageSnapshot))
                    }
                }
            }
            guard scheduleWake, !controls.isEmpty, !watchdogWakeQueued else { return false }
            watchdogWakeQueued = true
            return true
        }
        callback?(elapsed)
        for (control, usage) in terminals {
            // No row/page/refund mutation on this thread. The retirement handle
            // stays unacknowledged until the owning engine queue actually drains.
            control.output.finish(reason: .terminal(cause: .watchdog,
                message: CBv2TerminalCause.watchdog.diagnostic), usage: usage)
        }
        if wake {
            queue.async { [weak self] in
                guard let self else { return }
                self.lock.withLock { self.watchdogWakeQueued = false }
                // All rows may be paused, so schedulePump's runnable predicate
                // alone cannot drive lease expiry. Scan before asking it to run.
                self.expireLeases()
                self.schedulePump()
            }
        }
    }

    deinit { watchdogTimer?.cancel() }
}
