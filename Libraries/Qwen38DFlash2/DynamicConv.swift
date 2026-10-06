// Copyright 2026 bstnxbt
// SPDX-License-Identifier: Apache-2.0
// Swift port of dflash-mlx DFlash2 at the revision recorded in NOTICE.

import MLX
import MLXNN

func groupedDynamicCausalConvolution(
    hidden: MLXArray,
    dynamic: MLXArray,
    base: MLXArray,
    groupSize: Int
) -> MLXArray {
    let batch = hidden.dim(0)
    let length = hidden.dim(1)
    let hiddenSize = hidden.dim(2)
    let groups = hiddenSize / groupSize
    let blocks = hidden.reshaped([batch, length, groups, groupSize])
    var output = MLXArray.zeros(like: blocks)

    for offset in 0 ..< base.dim(0) {
        let values: MLXArray
        if offset == 0 {
            values = blocks
        } else {
            values = concatenated(
                [
                    MLXArray.zeros(
                        [batch, offset, groups, groupSize], dtype: hidden.dtype),
                    blocks[0..., 0 ..< (length - offset), 0..., 0...],
                ],
                axis: 1)
        }
        let fixed = base[offset, 0...]
            .reshaped([1, 1, groups, groupSize])
            .asType(hidden.dtype)
        let generated = dynamic[0..., 0..., offset, 0...]
            .expandedDimensions(axis: -1)
        output = output + fixed * values + generated * values
    }
    return output.reshaped(hidden.shape)
}

final class GroupedDynamicCausalConv: Module {
    let kernelSize: Int
    let groupSize: Int

    @ParameterInfo(key: "base_kernel") var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear

    init(hiddenSize: Int, kernelSize: Int, groupSize: Int) {
        self.kernelSize = kernelSize
        self.groupSize = groupSize
        let groups = hiddenSize / groupSize
        _baseKernel.wrappedValue = MLXArray.zeros([2, kernelSize, hiddenSize])
        _kernelProjection.wrappedValue = Linear(
            hiddenSize, 2 * kernelSize * groups, bias: false)
        super.init()
    }

    func prepare(_ hidden: MLXArray) -> (MLXArray, MLXArray) {
        let groups = hidden.dim(-1) / groupSize
        let dynamic = kernelProjection(hidden).reshaped(
            hidden.shape.dropLast() + [2, kernelSize, groups])
        return (
            groupedDynamicCausalConvolution(
                hidden: hidden,
                dynamic: dynamic[0..., 0..., 0, 0..., 0...],
                base: baseKernel[0, 0..., 0...],
                groupSize: groupSize),
            dynamic[0..., 0..., 1, 0..., 0...]
        )
    }

    func finish(_ hidden: MLXArray, dynamic: MLXArray) -> MLXArray {
        groupedDynamicCausalConvolution(
            hidden: hidden,
            dynamic: dynamic,
            base: baseKernel[1, 0..., 0...],
            groupSize: groupSize)
    }
}
