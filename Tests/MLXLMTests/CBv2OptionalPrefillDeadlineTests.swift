import MLX
import Testing

@testable import MLXLMCommon

@Suite("Optional prefill deadline arrivals", .serialized)
struct CBv2OptionalPrefillDeadlineTests {
    private func withFixture(
        partial: Bool = false, mode: PagedQuantizedPrefillMode = .opportunisticSDPA,
        _ body: (CBv2OptionalPrefillDeadlineFixture) async throws -> Void
    ) async throws {
        let fixture = try CBv2OptionalPrefillDeadlineFixture(partial: partial, mode: mode)
        do { try await body(fixture) }
        catch { await fixture.close(); throw error }
        await fixture.close()
    }

    @Test(arguments: [false, true])
    func arrivalReclaimsActualOptionalBytesAndKeepsRemainingWork(partial: Bool) async throws {
        try await withFixture(partial: partial) { fixture in
            let oldStream = try await fixture.holdFirstStep()
            let before = fixture.engine.capacity()
            let optionalBytes = fixture.backend.pool.quantizedPrefillStatistics.currentAdditionalWorkspaceBytes
            let original = fixture.policy()
            let result = try await fixture.engine.submit(fixture.target, firstTokenDeadline: original)
            guard case .admitted(let stream, let work, let admittedAt, _) = result,
                case .bounded(let scheduled, _) = work else {
                Issue.record("the unchanged raw request should fit after the owning step retires")
                return
            }
            #expect(admittedAt < original.deadline)
            #expect(scheduled.prefillTokens == fixture.target.promptTokens.count + (partial ? 33 : 0),
                "completed first chunk must not be forecast twice; unfinished prompt work remains")
            fixture.engine.loopForTesting.onEngineQueueSync {
                #expect(fixture.engine.loopForTesting.stepCount == 1)
                #expect(fixture.model.widths == [33], "draining must not construct a successor forward")
                #expect(fixture.backend.pool.quantizedPrefillStatistics.currentAdditionalWorkspaceBytes == 0)
                #expect(fixture.engine.capacity().kvBytesReserved <= before.kvBytesReserved - optionalBytes)
                if partial {
                    #expect(fixture.engine.capacity().kvBytesReserved == before.kvBytesReserved - optionalBytes,
                        "the existing raw request and other owners remain charged; only O was released")
                }
                #expect(fixture.engine.capacity().waitingRequests == 1)
                #expect(fixture.targetFits(), "the actual ledger, not a subtracted forecast, must now fit")
            }
            fixture.resume()
            let old = await cbv2SchedCollect(oldStream), next = await cbv2SchedCollect(stream)
            #expect(old.finishReason == .length && old.tokens.count == (partial ? 2 : 1))
            #expect(next.finishReason == .length && next.tokens == [1])
            fixture.engine.loopForTesting.onEngineQueueSync {
                #expect(fixture.model.widths.filter { $0 == 33 }.count == (partial ? 2 : 1))
            }
        }
    }

    @Test("time spent retiring the old step consumes the original absolute deadline")
    func drainElapsedTimeCannotRefreshDeadline() async throws {
        try await withFixture { fixture in
            let oldStream = try await fixture.holdFirstStep()
            fixture.engine.loopForTesting.onEngineQueueSync {
                // Confirmation occurs inside real finalize, after the GPU and
                // optional lease retire, but before the new admission verdict.
                fixture.sampler.onFirstConfirmation = { fixture.clock.advance(seconds: 2) }
            }
            let original = fixture.policy(seconds: 1)
            let result = try await fixture.engine.submit(fixture.target, firstTokenDeadline: original)
            guard case .deadlineUnreachable(let work) = result,
                case .bounded(let scheduled, _) = work else {
                Issue.record("memory must become reachable while the original time budget expires")
                return
            }
            #expect(scheduled.prefillTokens == 9)
            #expect(fixture.clock.clock.now() > original.deadline)
            fixture.engine.loopForTesting.onEngineQueueSync {
                #expect(fixture.engine.loopForTesting.scheduler.record(for: fixture.target.id) == nil)
                #expect(fixture.model.widths == [33])
                #expect(fixture.backend.pool.quantizedPrefillStatistics.currentAdditionalWorkspaceBytes == 0)
            }
            let old = await cbv2SchedCollect(oldStream)
            #expect(old.finishReason == .length && old.tokens == [1])
        }
    }

    @Test("cancellation during retirement holds admission identity until queue acknowledgement")
    func cancelledArrivalDoesNotCommitOrReleaseIdentityEarly() async throws {
        try await withFixture { fixture in
            let oldStream = try await fixture.holdFirstStep()
            let gate = CBv2OptionalDeadlineGate()
            fixture.gate = gate
            fixture.engine.loopForTesting.onEngineQueueSync {
                fixture.sampler.onFirstConfirmation = { gate.wait() }
            }
            let original = fixture.policy()
            let task = Task { try await fixture.engine.submit(fixture.target, firstTokenDeadline: original) }
            #expect(await cbv2SchedWait { gate.entered })
            task.cancel()
            #expect(throws: CBv2SchedulerError.self) { try fixture.engine.submit(fixture.target) }
            gate.release()
            do {
                _ = try await task.value
                Issue.record("cancelled deadline arrival must not commit")
            } catch is CancellationError {
                // The queued operation has now acknowledged its generation.
            }
            #expect(fixture.engine.loopForTesting.stream(for: fixture.target.id) == nil)
            let retry = try fixture.engine.submit(fixture.target)
            fixture.resume()
            let old = await cbv2SchedCollect(oldStream), next = await cbv2SchedCollect(retry)
            #expect(old.finishReason == .length && old.tokens == [1])
            #expect(next.finishReason == .length && next.tokens == [1])
        }
    }

    @Test("pending blocker cancellation is processed before optional-step finalization")
    func cancelledBlockerDoesNotEmitItsPendingSample() async throws {
        try await withFixture { fixture in
            let oldStream = try await fixture.holdFirstStep()
            fixture.engine.cancel(fixture.blocker.id)
            let result = try await fixture.engine.submit(fixture.target, firstTokenDeadline: fixture.policy())
            guard case .admitted(let stream, _, _, _) = result else {
                Issue.record("new arrival should fit after cancellation-owned fast step retirement")
                return
            }
            fixture.resume()
            let old = await cbv2SchedCollect(oldStream), next = await cbv2SchedCollect(stream)
            #expect(old.finishReason == .cancelled && old.tokens.isEmpty)
            #expect(next.finishReason == .length && next.tokens == [1])
        }
    }

    @Test("default direct mode keeps the existing in-flight deadline forecast behavior")
    func directModeDoesNotForceRetirement() async throws {
        try await withFixture(mode: .direct) { fixture in
            let oldStream = try await fixture.holdFirstStep(fillFreeGrant: false)
            let result = try await fixture.engine.submit(fixture.target, firstTokenDeadline: fixture.policy())
            guard case .admitted(let stream, _, _, _) = result else {
                Issue.record("direct-mode control should fit without draining its step")
                return
            }
            fixture.engine.loopForTesting.onEngineQueueSync {
                let old = fixture.engine.loopForTesting.scheduler.record(for: fixture.blocker.id)
                #expect(old?.generatedTokenCount == 0 && old?.pendingSamples == 1)
                #expect(fixture.engine.loopForTesting.stepCount == 1)
                #expect(fixture.backend.pool.quantizedPrefillStatistics.currentAdditionalWorkspaceBytes == 0)
            }
            fixture.resume()
            let old = await cbv2SchedCollect(oldStream), next = await cbv2SchedCollect(stream)
            #expect(old.finishReason == .length && old.tokens == [1])
            #expect(next.finishReason == .length && next.tokens == [1])
        }
    }
}
