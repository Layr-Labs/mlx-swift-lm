// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXVLM

extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextVisionLinearTests {
        @Test(
            .enabled(
                if: GLM5NextReferenceFixture.enabled,
                "Requires the unchanged canonical official CPU oracle"))
        func pinnedBiasedProjectionsMatchOfficial() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextReferenceFixture()
                var checked = 0
                for key in fixture.weights.keys.sorted()
                where key.hasPrefix("model.visual.") && key.hasSuffix(".weight") {
                    let weight = fixture.weights[key]!
                    guard weight.ndim == 2 else { continue }
                    let path = String(key.dropLast(7))
                    guard let bias = fixture.weights[path + ".bias"] else { continue }
                    let layer = GLM5NextVisionLinear(weight.dim(1), weight.dim(0), bias: true)
                    try layer.update(
                        parameters: ModuleParameters.unflattened([
                            ("weight", weight), ("bias", bias),
                        ]), verify: .all)
                    for name in ["image", "video"] {
                        let input = try fixture.tensor(name + "." + path + ".call0.input.0")
                        try MLX.withError { error in
                            let output = layer(input)
                            eval(output)
                            try error.check()
                            try fixture.compare(
                                output, key: name + "." + path + ".call0.output",
                                absolute: 0, relative: 0)
                        }
                        checked += 1
                    }
                }
                #expect(checked == 20)
            }
        }

        @Test(arguments: [DeviceType.cpu, .gpu])
        func outputWidthsDtypesAndStridedInputs(device: DeviceType) throws {
            try Device.withDefaultDevice(Device(device)) {
                for channels in [16, 17, 31, 32, 33, 48, 64] {
                    let rows = 7
                    let columns = 11
                    let x = (0 ..< (rows * columns)).map { Float($0 % 7 - 3) / 16 }
                    let w = (0 ..< (channels * columns)).map { Float($0 % 5 - 2) / 32 }
                    for hasBias in [false, true] {
                        let b = (0 ..< channels).map { hasBias ? Float($0 % 9 - 4) / 32 : 0 }
                        let expected = (0 ..< (rows * channels)).map { index -> Float in
                            var value = Double(b[index % channels])
                            for i in 0 ..< columns {
                                value +=
                                    Double(x[(index / channels) * columns + i])
                                    * Double(w[(index % channels) * columns + i])
                            }
                            return Float(value)
                        }
                        for dtype: DType in [.float32, .float16, .bfloat16] {
                            let layer = GLM5NextVisionLinear(columns, channels, bias: hasBias)
                            var parameters = [
                                ("weight", MLXArray(w, [channels, columns]).asType(dtype))
                            ]
                            if hasBias { parameters.append(("bias", MLXArray(b).asType(dtype))) }
                            try layer.update(
                                parameters: ModuleParameters.unflattened(parameters), verify: .all)
                            let input = MLXArray(x, [rows, columns]).asType(dtype)
                            let strided = stacked([input, input], axis: -1)
                                .reshaped(rows, columns * 2)[0..., .stride(by: 2)]
                            try MLX.withError { error in
                                let output = layer(strided)
                                let reference = MLXArray(expected, [rows, channels]).asType(dtype)
                                eval(output, reference)
                                StreamOrDevice.default.stream.synchronize()
                                try error.check()
                                #expect(output.shape == [rows, channels] && output.dtype == dtype)
                                #expect(
                                    output.asType(.float32).asArray(Float.self)
                                        == reference.asType(.float32).asArray(Float.self))
                            }
                        }
                    }
                }
            }
        }
    }
}
