// Copyright © 2026 Eigen Labs.

import MLX
import MLXLMCommon
import MLXNN

// Exact verify windows for the native tower. Exactness follows the row's role,
// not the call's row count. With them on, every projection and block norm of a
// decode or verify forward gives each row the bytes that row gets alone, at any
// row count, so one forward over a draft window equals the decode steps it
// replaces and a decode step's bytes do not depend on how many rows share it.
// Prompt chunks keep MLX's arithmetic: the engine runs each in its own
// forward, never beside decode rows. Off by default: the tower then keeps
// MLX's arithmetic everywhere.

/// The tower's numerics switch, shared by every block of one model.
final class MuseGlimmerNumerics {
    /// The row-exact products take 1 to 16 rows per call; larger calls run
    /// in pieces of this many rows.
    static let pieceRows = 16
    /// MLX's row reduction switches kernels at 32 rows (`row_reduce_simple`).
    static let reductionPieceFrom = 32

    var exactWindows = false
    /// Set while a prompt chunk's graph is built (`prompt(_:)`).
    private(set) var buildingPrompt = false

    static func rows(_ x: MLXArray) -> Int { x.dim(-1) == 0 ? 0 : x.size / x.dim(-1) }

    /// A decode or verify call of an exact tower, at any row count.
    func isExact(_ x: MLXArray) -> Bool { exactWindows && !buildingPrompt && Self.rows(x) > 0 }

    /// `body` built as a prompt chunk: its projections and norms keep MLX's
    /// arithmetic, exact windows on or off.
    func prompt<T>(_ body: () throws -> T) rethrows -> T {
        let outer = buildingPrompt
        buildingPrompt = true
        defer { buildingPrompt = outer }
        return try body()
    }

    /// `layer(x)`, through the row-exact product on an exact call.
    func project(_ layer: any UnaryLayer, _ x: MLXArray) -> MLXArray {
        guard isExact(x) else { return layer(x) }
        guard let exact = layer as? any MuseGlimmerRowExactProjection,
            exact.supportsRowExactWindows(dtype: x.dtype)
        else { preconditionFailure("Muse Glimmer projection without a row-exact path") }
        return Self.inPieces(x, rows: Self.pieceRows) { exact.rowExactProduct($0) }
    }

    /// `body(x)` on pieces of at most `rows` rows, joined in order.
    static func inPieces(_ x: MLXArray, rows: Int, _ body: (MLXArray) -> MLXArray) -> MLXArray {
        let count = Self.rows(x)
        guard count > rows else { return body(x) }
        let width = x.dim(-1)
        let flat = x.reshaped(count, width)
        let pieces = stride(from: 0, to: count, by: rows).map {
            body(flat[$0 ..< min(count, $0 + rows)])
        }
        let joined = concatenated(pieces, axis: 0)
        return joined.reshaped(Array(x.shape.dropLast()) + [joined.dim(-1)])
    }

    /// A row-reducing region (`body`) kept below MLX's reduction switch on
    /// an exact call, so each row reduces as it does alone.
    func reduceRows(_ inputs: [MLXArray], _ body: ([MLXArray]) -> MLXArray) -> MLXArray {
        let x = inputs[0]
        guard isExact(x), Self.rows(x) >= Self.reductionPieceFrom else { return body(inputs) }
        let count = Self.rows(x)
        let flat = inputs.map { $0.reshaped(count, $0.dim(-1)) }
        let pieces = stride(from: 0, to: count, by: Self.pieceRows).map { start in
            body(flat.map { $0[start ..< min(count, start + Self.pieceRows)] })
        }
        let joined = concatenated(pieces, axis: 0)
        return joined.reshaped(Array(x.shape.dropLast()) + [joined.dim(-1)])
    }
}

/// A projection slot with a product whose rows do not depend on the row count.
protocol MuseGlimmerRowExactProjection: UnaryLayer {
    func supportsRowExactWindows(dtype: DType) -> Bool
    /// 1 to 16 rows; each row equals the one-row call.
    func rowExactProduct(_ x: MLXArray) -> MLXArray
}

extension ModelOptWeightOnlyNVFP4Linear: MuseGlimmerRowExactProjection {
    func supportsRowExactWindows(dtype: DType) -> Bool {
        ModelOptWeightOnlyNVFP4RowExact.supports(rows: 1, inputs: inputDimensions, dtype: dtype)
    }

    /// One order of operations at every row count, the one-row step included.
    func rowExactProduct(_ x: MLXArray) -> MLXArray {
        ModelOptWeightOnlyNVFP4RowExact.product(
            x, weight: weight, blockScales: weightScale, tensorScale: weightScale2)
    }
}

extension ModelOptCalibratedLinear: MuseGlimmerRowExactProjection {
    func supportsRowExactWindows(dtype: DType) -> Bool {
        format == .fp8 && execution == .graph && inputDimensions > 64 && weightDType == dtype
    }

    /// One row is the graph path's vector product; 2 to 16 rows the row-exact
    /// product over the same tiles.
    func rowExactProduct(_ x: MLXArray) -> MLXArray {
        do {
            return MuseGlimmerNumerics.rows(x) == 1
                ? try graphForward(x) : try rowExactForward(x)
        } catch {
            preconditionFailure("Invalid calibrated ModelOpt projection: \(error)")
        }
    }
}

extension Linear: MuseGlimmerRowExactProjection {
    func supportsRowExactWindows(dtype: DType) -> Bool {
        type(of: self) == Linear.self && bias == nil && weight.ndim == 2
            && weight.dim(1) > 64 && weight.dtype == dtype
    }

    /// One row is MLX's single-vector product; more rows reproduce it per row.
    func rowExactProduct(_ x: MLXArray) -> MLXArray {
        let rows = MuseGlimmerNumerics.rows(x)
        guard rows > 1 else { return self(x) }
        let product = ModelOptRowExact.matmul(x.reshaped(rows, x.dim(-1)), weight)
        return product.reshaped(Array(x.shape.dropLast()) + [weight.dim(0)])
    }
}

public enum MuseGlimmerRowExactError: Error, Equatable {
    /// The projection at this model path has no row-exact product.
    case unsupportedProjection(String)
}

extension MuseGlimmerTextModel {
    /// Whether decode and verify rows are row-exact: each row's projections
    /// and norms are those of the one-row call, whatever the row count.
    /// Prompt chunks keep MLX's arithmetic either way.
    public var verifiesWindowsExactly: Bool { numerics.exactWindows }

    /// A name for the numerics, for the checkpoint numerics fingerprint:
    /// decode bytes change when exact windows are on. `v1` ran every call of
    /// up to 64 rows exact, prompt chunks included; `v2` ran every call
    /// exact; `v3` runs decode and verify rows exact and prompt chunks as MLX
    /// does.
    public var numericsProfile: String {
        numerics.exactWindows ? "muse-glimmer-exact-windows-v3" : "mlx"
    }

    /// The model path and layer of every projection slot.
    var projectionSlots: [(path: String, layer: Module)] {
        leafModules().flattened().filter { _, module in
            module is Linear || module is ModelOptCalibratedLinear
                || module is ModelOptWeightOnlyNVFP4Linear
        }
    }

    /// Turns exact windows on or off for the whole tower. Every projection must
    /// have a row-exact product for the activation dtype; otherwise this throws
    /// and leaves the tower as it was. On, FP8 products take the window schedule
    /// (the same bytes, scheduled for windows). Set before serving; not while a forward
    /// is being built.
    public func setRowExactWindows(_ enabled: Bool) throws {
        let dtype = activationDType
        let slots = projectionSlots
        if enabled {
            for (path, layer) in slots {
                guard let exact = layer as? any MuseGlimmerRowExactProjection,
                    exact.supportsRowExactWindows(dtype: dtype)
                else { throw MuseGlimmerRowExactError.unsupportedProjection(path) }
            }
        }
        for case (_, let layer as ModelOptCalibratedLinear) in slots {
            layer.rowExactSchedule = enabled ? .windows : .standard
        }
        numerics.exactWindows = enabled
    }
}
