import MLX

/// What one donor can give up so a candidate fits under the slot-wide cap.
struct CBv2HistoricalStagingAllowance {
    let requestID: CBv2RequestID
    /// Window bytes of the donor's role-less rolling latest, which the
    /// candidate's commit retires anyway.
    var replacingBytes = 0
    /// The donor's own staged boundaries the candidate may displace, lowest
    /// priority first.
    var sheddable: [Int] = []
}

extension CBv2CompleteCheckpointCapture {
    /// Give up staged boundaries of one donor. Whenever the candidate that
    /// displaced them then fails to commit, its step was discarded and the
    /// donor's whole staged set is dropped with it.
    func historicalStagingSheddable(_ allowance: CBv2HistoricalStagingAllowance?)
        -> [CBv2CapturedCompleteCheckpoint]
    {
        guard let allowance, let captures = staged[allowance.requestID] else { return [] }
        return allowance.sheddable.compactMap { position in
            captures.first { $0.position == position }
        }
    }

    func releaseHistoricalStaging(
        _ captures: [CBv2CapturedCompleteCheckpoint], requestID: CBv2RequestID
    ) {
        let positions = Set(captures.compactMap(\.position))
        guard !positions.isEmpty else { return }
        staged[requestID]?.removeAll { $0.position.map(positions.contains) ?? false }
        for position in positions { retentions[requestID]?.shed(position) }
        if staged[requestID]?.isEmpty == true { staged.removeValue(forKey: requestID) }
        for capture in captures { retireCaptured(capture, requestID: requestID) }
    }

    func historicalStagingAllowance(
        requestID: CBv2RequestID,
        retention: CBv2CheckpointRetention,
        role: CBv2HistoricalStagingCap.Role
    )
        -> CBv2HistoricalStagingAllowance
    {
        let current = retentions[requestID] ?? retention
        var result = CBv2HistoricalStagingAllowance(
            requestID: requestID, sheddable: current.sheddable(for: role))
        if role == .latest, let replaced = current.replaceableLatest {
            result.replacingBytes =
                staged[requestID]?
                .first { $0.position == replaced }?.stagedHistoricalBytes ?? 0
        }
        return result
    }

    /// Shared retention operates only on successfully captured positions.
    /// Native callers supply no stride; neither the hint nor a resumed donor
    /// fabricates target/assistant state at an unobserved frontier.
    func stageHistorical(
        _ candidate: CBv2CapturedCompleteCheckpoint,
        requestID: CBv2RequestID, position: Int, stride: Int?,
        hintTokens: Int?, resumedAt: Int
    )
        -> [CBv2CapturedCompleteCheckpoint]
    {
        var retention = retention(
            requestID: requestID, stride: stride,
            hintTokens: hintTokens, resumedAt: resumedAt)
        var checkpoints = (staged[requestID] ?? []) + [candidate]
        let bytesByPosition = Dictionary(
            checkpoints.map { ($0.position ?? 0, $0.stagedHistoricalBytes) },
            uniquingKeysWith: { $0 + $1 })
        let retired = retention.commitHistoricalBounded(
            position,
            bytesByPosition: bytesByPosition, byteBudget: historicalStagedByteBudget)
        let retiring = checkpoints.filter { retired.contains($0.position ?? 0) }
        checkpoints.removeAll { retired.contains($0.position ?? 0) }
        if checkpoints.isEmpty {
            staged.removeValue(forKey: requestID)
            retentions.removeValue(forKey: requestID)
        } else {
            staged[requestID] = checkpoints
            retentions[requestID] = retention
        }
        return retiring
    }

}

extension CBv2CheckpointRetention {
    /// The slot cap refuses copies before construction. At commit, this
    /// existing per-donor policy sheds first, then target under byte pressure,
    /// keeping the rolling latest that already passed the slot/ledger bounds.
    mutating func commitHistoricalBounded(
        _ position: Int, bytesByPosition: [Int: Int],
        byteBudget: Int
    ) -> Set<Int> {
        var retired = Set(commit(position))
        var bytes = retained.reduce(0) { $0 + (bytesByPosition[$1] ?? 0) }
        for shed in sheddable(for: .latest)
        where bytes > byteBudget && shed != retained.last {
            self.shed(shed)
            retired.insert(shed)
            bytes -= bytesByPosition[shed] ?? 0
        }
        return retired
    }
}
