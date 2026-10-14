// Gated-delta parity: the backend kernel (gatedDeltaKernel through MLXShim)
// against gatedDeltaOps, on the shapes of the Layr-Labs mlx-shim CUDA
// gated-delta test: B=1, Hk=16, Hv=32, Dk=128, Dv=128, T in {1, 7, 128,
// 2048}, and Dv=40 at T=7. Thresholds as there: y max abs 2e-2, fp32 state
// max abs 1e-3.
//
// Build with `-Xswiftc -enable-testing` (it uses internal MLXLLM functions).

import Foundation
import MLX
import MLXShim

@testable import MLXLLM

@main
enum GDNParity {
    static let B = 1
    static let Hk = 16
    static let Hv = 32
    static let Dk = 128
    static let Dv = 128

    struct Inputs {
        let q, k, v, g, beta, state: MLXArray
    }

    static func e(_ x: Float) -> String { String(format: "%.4e", Double(x)) }

    static func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    static func maxRel(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = abs(a.asType(.float32) - b.asType(.float32))
        let den = maximum(abs(b.asType(.float32)), MLXArray(Float(1e-8)))
        return (d / den).max().item(Float.self)
    }

    static func allFinite(_ arrays: [MLXArray]) -> Bool {
        arrays.allSatisfy { isFinite($0.asType(.float32)).all().item(Bool.self) }
    }

    // mlx-lm normalize_qk: q * inv_scale^2 * rms_norm(q), k * inv_scale * rms_norm(k).
    static func rmsNormNoWeight(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
        let x32 = x.asType(.float32)
        return x32 * rsqrt(mean(x32 * x32, axis: -1, keepDims: true) + eps)
    }

    static func makeInputs(T: Int, hv: Int, dv: Int, seed: UInt64) -> Inputs {
        MLXRandom.seed(seed)
        let invScale = Float(1.0 / Double(Dk).squareRoot())
        let q0 = MLXRandom.normal([B, T, Hk, Dk]).asType(.bfloat16)
        let k0 = MLXRandom.normal([B, T, Hk, Dk]).asType(.bfloat16)
        let v = MLXRandom.normal([B, T, hv, dv]).asType(.bfloat16)
        let q = (rmsNormNoWeight(q0) * (invScale * invScale)).asType(.bfloat16)
        let k = (rmsNormNoWeight(k0) * invScale).asType(.bfloat16)
        let g = MLXRandom.uniform(low: Float(0.5), high: Float(1.0), [B, T, hv])
        let beta = MLXRandom.uniform(low: Float(0.0), high: Float(1.0), [B, T, hv])
        let state = MLXRandom.normal([B, hv, dv, Dk])
        eval(q, k, v, g, beta, state)
        return Inputs(q: q, k: k, v: v, g: g, beta: beta, state: state)
    }

    /// Returns true on pass.
    static func runCase(T: Int, hv: Int, dv: Int, label: String) -> Bool {
        let x = makeInputs(T: T, hv: hv, dv: dv, seed: UInt64(T))
        let (yRef, sRef) = gatedDeltaOps(
            q: x.q, k: x.k, v: x.v, g: x.g, beta: x.beta, state: x.state)
        let (yK, sK) = gatedDeltaKernel(
            q: x.q, k: x.k, v: x.v, g: x.g, beta: x.beta, state: x.state)
        eval(yRef, sRef, yK, sK)
        let ya = maxAbs(yK, yRef)
        let sa = maxAbs(sK, sRef)
        let finite = allFinite([yRef, sRef, yK, sK])
        let ok = finite && ya <= 2e-2 && sa <= 1e-3 && sK.dtype == .float32
        print(
            "\(ok ? "PASS" : "FAIL") \(label): y abs=\(e(ya)) rel=\(e(maxRel(yK, yRef)))"
                + "  state abs=\(e(sa)) rel=\(e(maxRel(sK, sRef)))  finite=\(finite)")
        return ok
    }

    /// Masked input: the masked kernel has no CUDA body, so gatedDeltaUpdate
    /// must take gatedDeltaOps and match it bit for bit off Metal.
    static func maskedCase() -> Bool {
        let T = 7
        MLXRandom.seed(5)
        let q = MLXRandom.normal([B, T, Hk, Dk]).asType(.bfloat16)
        let k = MLXRandom.normal([B, T, Hk, Dk]).asType(.bfloat16)
        let v = MLXRandom.normal([B, T, Hv, Dv]).asType(.bfloat16)
        let a = MLXRandom.normal([B, T, Hv]).asType(.bfloat16)
        let b = MLXRandom.normal([B, T, Hv]).asType(.bfloat16)
        let aLog = MLXRandom.normal([Hv])
        let dtBias = MLXRandom.normal([Hv])
        let state = MLXRandom.normal([B, Hv, Dv, Dk])
        let mask = MLXRandom.uniform(low: Float(0), high: Float(1), [B, T]) .> Float(0.3)
        let (yU, sU) = gatedDeltaUpdate(
            q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: state, mask: mask)
        let g = computeGatedDeltaG(aLog, a, dtBias)
        let beta = sigmoid(b).asType(.float32)
        let (yO, sO) = gatedDeltaOps(
            q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
        eval(yU, sU, yO, sO)
        let ya = maxAbs(yU, yO)
        let sa = maxAbs(sU, sO)
        let expectExact = ComputeBackend.current != .metal
        let ok = !expectExact || (ya == 0 && sa == 0)
        print(
            "\(ok ? "PASS" : "FAIL") masked T=7 gatedDeltaUpdate vs gatedDeltaOps:"
                + " y abs=\(e(ya)) state abs=\(e(sa))\(expectExact ? " (expect 0: op path)" : "")")

        let stacked = gatedDeltaUpdateStacked(
            q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: state)
        print("stacked kernel: \(stacked == nil ? "nil (declines)" : "present")")
        return ok
    }

    static func medianMs(_ body: () -> (MLXArray, MLXArray)) -> Double {
        for _ in 0 ..< 3 {
            let (y, s) = body()
            eval(y, s)
        }
        var times = [Double]()
        for _ in 0 ..< 10 {
            let t0 = DispatchTime.now().uptimeNanoseconds
            let (y, s) = body()
            eval(y, s)
            times.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
        }
        times.sort()
        return (times[4] + times[5]) / 2
    }

    static func timing(T: Int, seed: UInt64) {
        let x = makeInputs(T: T, hv: Hv, dv: Dv, seed: seed)
        let ops = medianMs {
            gatedDeltaOps(q: x.q, k: x.k, v: x.v, g: x.g, beta: x.beta, state: x.state)
        }
        let kernel = medianMs {
            gatedDeltaKernel(q: x.q, k: x.k, v: x.v, g: x.g, beta: x.beta, state: x.state)
        }
        print(
            "timing median ms T=\(T): ops=\(String(format: "%.3f", ops))"
                + " kernel=\(String(format: "%.3f", kernel))")
    }

    static func main() {
        print("backend: \(ComputeBackend.current.rawValue)")
        print("shapes: B=\(B) Hk=\(Hk) Hv=\(Hv) Dk=\(Dk) Dv=\(Dv); Dv=40 case at T=7")
        var ok = true
        for T in [1, 7, 128, 2048] {
            ok = runCase(T: T, hv: Hv, dv: Dv, label: "T=\(T) state=float32") && ok
        }
        ok = runCase(T: 7, hv: Hv, dv: 40, label: "T=7 state=float32 Dv=40") && ok
        ok = maskedCase() && ok
        timing(T: 1, seed: 99)
        timing(T: 2048, seed: 100)
        print(ok ? "GDN_PARITY PASS" : "GDN_PARITY FAIL")
        fflush(nil)  // all streams; Swift 6 rejects a reference to the C global stdout
        // _exit skips the exit handlers: MLX core cb77239 aborts in the static
        // CUDA CommandEncoder destructor during exit (device.cpp:216).
        _exit(ok ? 0 : 1)
    }
}
