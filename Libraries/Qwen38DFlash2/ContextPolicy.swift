// Copyright 2026 Youssof Altoukhi
// SPDX-License-Identifier: Apache-2.0
// Swift port of the Qwen 3.8 DFlash2 policy at the revision in NOTICE.

import Foundation

public struct DFlash2ContextPolicy: Sendable {
    private static let longContextBoundary = 16_384
    private static let fullPhysicalWidth = 8
    private static let headStepCostRatio = 0.20
    private static let updateAlpha = 0.15

    private let fixedM8: Bool
    private var positionAcceptanceEMA: [Double]
    private var draftDepth: Int

    public init(promptLength: Int) {
        fixedM8 = promptLength >= Self.longContextBoundary
        positionAcceptanceEMA = (0 ..< 7).map { 0.85 * pow(0.98, Double($0)) }
        draftDepth = 0
        if !fixedM8 {
            draftDepth = Self.recomputedDepth(positionAcceptanceEMA)
        }
    }

    public var nextPhysicalWidth: Int {
        fixedM8 ? Self.fullPhysicalWidth : max(1, min(8, 1 + draftDepth))
    }

    public mutating func record(blockLength: Int, acceptedDraftTokens: Int) {
        guard !fixedM8 else { return }
        let physicalWidth = max(1, min(Self.fullPhysicalWidth, blockLength))
        let attempted = physicalWidth - 1
        let accepted = max(0, min(acceptedDraftTokens, attempted))
        guard attempted > 0 else { return }

        for index in 0 ..< accepted {
            let value = positionAcceptanceEMA[index]
            positionAcceptanceEMA[index] = value + Self.updateAlpha * (1 - value)
        }
        if accepted < attempted {
            let value = positionAcceptanceEMA[accepted]
            positionAcceptanceEMA[accepted] = value - Self.updateAlpha * value
        } else if accepted < positionAcceptanceEMA.count {
            let value = positionAcceptanceEMA[accepted]
            positionAcceptanceEMA[accepted] = value + Self.updateAlpha * (1 - value)
        }
        draftDepth = Self.recomputedDepth(positionAcceptanceEMA)
    }

    private static func recomputedDepth(_ positionAcceptanceEMA: [Double]) -> Int {
        var reach = 1.0
        var expectedAccepted = 0.0
        var depth = 0
        while depth < 7 {
            reach *= positionAcceptanceEMA[depth]
            let threshold =
                headStepCostRatio * (1 + expectedAccepted)
                / (1 + Double(depth) * headStepCostRatio)
            if reach <= threshold { break }
            expectedAccepted += reach
            depth += 1
        }
        return depth
    }
}
