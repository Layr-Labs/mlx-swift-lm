// Adapted from MLX-VLM e79b0e041677ec4ca5333ba750376bb4e8c434cb.
// Copyright © 2025 Prince Canuma. MIT: docs/diffusiongemma/LICENSE-MLX-VLM.

import Foundation
import MLXLLM

/// Native aspect-preserving image/frame geometry. The actual grid determines
/// placeholder count; maxSoftTokens is a budget, not an output length.
public struct DiffusionGemmaMediaGeometry: Sendable, Equatable {
    public let width: Int
    public let height: Int
    public let softTokens: Int

    public static func resized(width: Int, height: Int, patchSize: Int,
        poolingSize: Int, maxSoftTokens: Int) throws -> Self
    {
        guard width > 0, height > 0, patchSize > 0, poolingSize > 0,
            [70, 140, 280, 560, 1120].contains(maxSoftTokens)
        else { throw DiffusionGemmaModelError.invalidInput("media geometry") }
        let sideProduct = patchSize.multipliedReportingOverflow(by: poolingSize)
        guard !sideProduct.overflow, sideProduct.partialValue > 0 else {
            throw DiffusionGemmaModelError.invalidInput("media geometry overflow")
        }
        let side = sideProduct.partialValue
        let maximum = side.multipliedReportingOverflow(by: maxSoftTokens)
        guard !maximum.overflow else {
            throw DiffusionGemmaModelError.invalidInput("media geometry overflow")
        }
        let targetPixels = Double(maxSoftTokens) * Double(side) * Double(side)
        let factor = sqrt(targetPixels / (Double(height) * Double(width)))
        var targetHeight = floor(factor * Double(height) / Double(side)) * Double(side)
        var targetWidth = floor(factor * Double(width) / Double(side)) * Double(side)
        guard targetHeight != 0 || targetWidth != 0 else {
            throw DiffusionGemmaModelError.invalidInput("empty resized media")
        }
        // Reference thin-image handling also clamps the long axis; simply
        // setting a zero dimension to one patch group can exceed the budget.
        if targetHeight == 0 {
            targetHeight = Double(side)
            targetWidth = min(floor(Double(width) / Double(height)) * Double(side), Double(maximum.partialValue))
        } else if targetWidth == 0 {
            targetWidth = Double(side)
            targetHeight = min(floor(Double(height) / Double(width)) * Double(side), Double(maximum.partialValue))
        }
        guard let h = Int(exactly: targetHeight), let w = Int(exactly: targetWidth),
            h > 0, w > 0, h.isMultiple(of: side), w.isMultiple(of: side)
        else { throw DiffusionGemmaModelError.invalidInput("resized media range") }
        let tokens = (h / side).multipliedReportingOverflow(by: w / side)
        guard !tokens.overflow, tokens.partialValue > 0, tokens.partialValue <= maxSoftTokens else {
            throw DiffusionGemmaModelError.invalidInput("resized media budget")
        }
        return Self(width: w, height: h, softTokens: tokens.partialValue)
    }
}
