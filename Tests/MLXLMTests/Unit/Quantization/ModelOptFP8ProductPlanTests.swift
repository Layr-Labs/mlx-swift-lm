import Testing

@testable import MLXLMCommon

extension UnitTests {

    /// The multi-row FP8 graph path's products near the eager reference's
    /// 64 MiB bound, where the eager tiles shrink toward one output row.
    @Suite
    struct ModelOptFP8ProductPlanTests {
        /// Super's FP8 projections (inputs, outputs): Mamba in_proj and
        /// out_proj, attention q/o and k/v.
        static let superShapes = [(4096, 18560), (8192, 4096), (4096, 4096), (4096, 256)]

        /// M3 Ultra, M4 Max, M5 Ultra and M5 Max routing.
        static let models = [
            ModelOptMLXProductModel(nax: false, size: "d"),
            ModelOptMLXProductModel(nax: false, size: "s"),
            ModelOptMLXProductModel(nax: true, size: "d"),
            ModelOptMLXProductModel(nax: true, size: "s"),
        ]

        private func plan(_ rows: Int, _ shape: (Int, Int), _ model: ModelOptMLXProductModel?)
            -> [ModelOptFP8ProductPiece]
        {
            ModelOptCalibratedLinear.fp8ProductPlan(
                rows: rows, inputs: shape.0, outputs: shape.1, model: model)
        }

        private func eager(_ rows: Int, _ shape: (Int, Int)) -> [Range<Int>] {
            ModelOptCalibratedLinear.fp8EagerTiles(rows: rows, inputs: shape.0, outputs: shape.1)
        }

        /// The stall's rows: each pole's eager tiling needs thousands of
        /// products; every plan needs at most 128.
        @Test(arguments: [
            ((4096, 18560), Array(82 ... 92)), ((8192, 4096), Array(165 ... 171)),
            ((4096, 4096), Array(250 ... 256)), ((4096, 256), Array(475 ... 482)),
        ])
        func productsNeverCollapseNearTheBound(shape: (Int, Int), rows: [Int]) {
            #expect(eager(92, (4096, 18560)).count == 6187)
            #expect(eager(170, (8192, 4096)).count == 4096)
            for model in Self.models + [nil] {
                for r in rows {
                    let pieces = plan(r, shape, model)
                    let dispatches = pieces.reduce(0) { $0 + $1.dispatches }
                    #expect(
                        dispatches <= ModelOptFP8ProductPiece.maximumDispatches,
                        "rows \(r) \(shape) \(String(describing: model)): \(dispatches) products")
                }
            }
        }

        /// Every row count of a prefill chunk: the pieces cover each output
        /// column once, in order, with at most 128 products.
        @Test func everyChunkCoversTheOutputsWithBoundedProducts() {
            for shape in Self.superShapes {
                for model in Self.models {
                    for rows in 2 ... 2048 {
                        let pieces = plan(rows, shape, model)
                        var next = 0
                        for piece in pieces {
                            #expect(piece.columns.lowerBound == next && !piece.columns.isEmpty)
                            next = piece.columns.upperBound
                            #expect(piece.rowBlocks.first?.lowerBound == 0)
                            #expect(piece.rowBlocks.last?.upperBound == rows)
                        }
                        #expect(next == shape.1)
                        #expect(
                            pieces.reduce(0) { $0 + $1.dispatches }
                                <= ModelOptFP8ProductPiece.maximumDispatches)
                    }
                }
            }
        }

        /// Wherever the eager reference runs, each product is in the class of
        /// the eager tiles that cover its columns, so its outputs keep their
        /// bits. The only exceptions are the NAX thin products of eager tiles
        /// narrower than 32 columns (MPP's masked path): those calls take
        /// slices. On M1 to M4 every Super call keeps its bits.
        @Test func piecesKeepTheEagerProductClass() {
            var sliced: [String: [Int]] = [:]
            for shape in Self.superShapes {
                for model in Self.models {
                    for rows in 16 ... 2048 {
                        let limit = (64 << 20) / 32
                        guard limit - rows * (shape.0 + shape.1) >= shape.0 else { continue }
                        let tiles = eager(rows, shape)
                        // Eager tiles share one width but the last.
                        func tile(_ column: Int) -> Range<Int> {
                            tiles[min(column / tiles[0].count, tiles.count - 1)]
                        }
                        let keeps = plan(rows, shape, model).allSatisfy { piece in
                            let expected = model.productClass(
                                rows: rows, inputs: shape.0,
                                outputs: tile(piece.columns.lowerBound).count)
                            let width = piece.vectors ? 1 : piece.columns.count
                            return model.productClass(
                                rows: rows, inputs: shape.0,
                                outputs: tile(piece.columns.upperBound - 1).count) == expected
                                && piece.rowBlocks.allSatisfy {
                                    model.productClass(
                                        rows: $0.count, inputs: shape.0, outputs: width)
                                        == expected
                                }
                        }
                        if !keeps {
                            #expect(
                                model.productClass(
                                    rows: rows, inputs: shape.0, outputs: tiles[0].count)
                                    == .thin(kGroups: 4, maskedRows: false, maskedColumns: true),
                                "rows \(rows) \(shape) \(model)")
                            sliced["\(model.nax) \(shape.0)x\(shape.1)", default: []].append(rows)
                        }
                    }
                }
            }
            #expect(sliced.keys.allSatisfy { $0.hasPrefix("true") })
            #expect(sliced["true 4096x18560"] == Array(87 ... 92) + Array(87 ... 92))
            #expect(sliced["true 8192x4096"] == Array(150 ... 169) + Array(150 ... 169))
            #expect(sliced["true 4096x4096"] == Array(241 ... 255) + Array(241 ... 255))
            #expect(sliced["true 4096x256"] == nil)
        }

        /// MLX 83dbdb04's routing at the stall's shapes.
        @Test func productClassesFollowMLXRouting() {
            let m3 = Self.models[0]
            let m5 = Self.models[2]
            #expect(m3.productClass(rows: 92, inputs: 4096, outputs: 3) == .splitK(partitions: 32))
            #expect(
                m3.productClass(rows: 92, inputs: 4096, outputs: 160) == .splitK(partitions: 32))
            #expect(
                m3.productClass(rows: 92, inputs: 4096, outputs: 161) == .splitK(partitions: 16))
            #expect(m3.productClass(rows: 92, inputs: 4096, outputs: 8192) == .regular)
            #expect(m3.productClass(rows: 170, inputs: 8192, outputs: 1) == .vector)
            #expect(m3.productClass(rows: 15, inputs: 4096, outputs: 512) == nil)
            #expect(
                m5.productClass(rows: 92, inputs: 4096, outputs: 3)
                    == .thin(kGroups: 4, maskedRows: false, maskedColumns: true))
            #expect(
                m5.productClass(rows: 86, inputs: 4096, outputs: 36)
                    == .thin(kGroups: 4, maskedRows: false, maskedColumns: false))
            #expect(
                m5.productClass(rows: 43, inputs: 4096, outputs: 8192)
                    == .thin(kGroups: 4, maskedRows: false, maskedColumns: false))
            #expect(
                m5.productClass(rows: 80, inputs: 4096, outputs: 69)
                    == .naxSplitK(partitionSize: 2048))
            #expect(m5.productClass(rows: 2048, inputs: 4096, outputs: 4096) == .regular)
            #expect(ModelOptMLXProductModel.nextPowerOfTwo(17) == 32)
            #expect(ModelOptMLXProductModel.nextPowerOfTwo(16) == 16)
            #expect(ModelOptMLXProductModel.nextPowerOfTwo(0) == 0)
        }

        /// Far from the bound, calls keep the eager tiles unchanged.
        @Test func callsAwayFromTheBoundKeepTheEagerTiles() {
            for shape in Self.superShapes {
                for rows in [2, 7, 16, 33] {
                    for model in Self.models + [nil] {
                        #expect(plan(rows, shape, model).map(\.columns) == eager(rows, shape))
                    }
                }
            }
        }
    }
}
