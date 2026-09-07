import Foundation
import MLX
import MLXFast

/// Shared packed format interpretation. Decode reconstructs only its register
/// tile; no allocation proportional to the sequence length is introduced here.
enum PagedQuantizedMetal {
    static let header = """
        #include <metal_stdlib>
        using namespace metal;
        namespace cbv2 {
        inline float quant_sign(uint index) {
            uint x = index + 0x9e3779b9u;
            x = (x ^ (x >> 16)) * 0x7feb352du;
            x = (x ^ (x >> 15)) * 0x846ca68bu;
            x ^= x >> 16;
            return (x & 1u) ? -1.0f : 1.0f;
        }
        template<int BITS, int D, int G>
        inline float quant_load(const device uchar* row, int column) {
            constexpr int DATA = D * BITS / 8;
            constexpr int GROUPS = D / G;
            const device float* scales = reinterpret_cast<const device float*>(row + DATA);
            const device float* offsets = scales + GROUPS;
            const uint code = BITS == 4
                ? (uint(row[column / 2]) >> ((column & 1) * 4)) & 15u
                : uint(row[column]);
            return scales[column / G] * float(code) + offsets[column / G];
        }
        template<typename T, int D, int G, int KB, int VB>
        struct PagedQuantizedKVPage {
            const device uchar* key;
            const device uchar* value;
            template<int EPT>
            void load(size_t base, uint lane, thread float* k, thread float* v) const {
                constexpr int KROW = D * KB / 8 + 8 * (D / G);
                constexpr int VROW = D * VB / 8 + 8 * (D / G);
                const size_t row = base / D;
                for (int e = 0; e < EPT; e++) {
                    const int column = lane * EPT + e;
                    k[e] = quant_load<KB, D, G>(key + row * KROW, column);
                    v[e] = quant_load<VB, D, G>(value + row * VROW, column);
                }
            }
            // Packed writes occur in a separately fenced token-local kernel.
            // This method only satisfies the native accessor's static interface.
            template<int WIDTH, int TG>
            void write(size_t, const device T*, const device T*, int) const {}
        };
        template<typename T, int N, int D, int S, int G, int KB, int VB>
        struct PagedQuantizedSegmentAccessor {
            const device uchar* buffers[N];
            const device int64_t* value_offsets;
            const device int32_t* record;
            int kv_heads;
            using Page = PagedQuantizedKVPage<T, D, G, KB, VB>;
            Page page(int binding, int local) const {
                constexpr int KROW = D * KB / 8 + 8 * (D / G);
                constexpr int VROW = D * VB / 8 + 8 * (D / G);
                const size_t kp = (size_t)kv_heads * S * KROW;
                const size_t vp = (size_t)kv_heads * S * VROW;
                if (binding < 0 || binding >= N || local < 0
                    || (size_t)local >= (size_t)value_offsets[binding] / kp) {
                    binding = 0; local = 0;
                }
                return {buffers[binding] + (size_t)local * kp,
                    buffers[binding] + value_offsets[binding] + (size_t)local * vp};
            }
            Page write_page(int) const { return page(record[4], record[5]); }
            Page read_page(const device int32_t*, int logical, int) const {
                int index = logical - record[2];
                if (index < 0 || index >= record[3]) return page(0, 0);
                return page(record[8 + 2 * index], record[9 + 2 * index]);
            }
        };
        }
        """

    /// Each threadgroup owns a complete head vector, so butterflies never
    /// cross a group barrier or depend on another token's transform.
    static let rotateBody = """
        const uint column = thread_position_in_threadgroup.x;
        const uint row = threadgroup_position_in_grid.y;
        threadgroup float scratch[D];
        float x = float(input[row * D + column]);
        scratch[column] = INVERSE ? x : x * cbv2::quant_sign(column % R);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1; stride < R; stride <<= 1) {
            const float a = scratch[column];
            const float b = scratch[column ^ stride];
            const float result = (column & stride) ? b - a : a + b;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            scratch[column] = result;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float result = scratch[column] * rsqrt(float(R));
        if (INVERSE) result *= cbv2::quant_sign(column % R);
        output[row * D + column] = T(result);
        """

    private static let lock = NSLock()
    nonisolated(unsafe) private static var rotations: [String: MLXFast.MLXFastKernel] = [:]

    static func rotate(_ input: MLXArray, config: PagedKVQuantizationConfig,
                       inverse: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        let d = input.dim(-1), r = config.resolvedRotationBlockSize(headDim: input.dim(-1))
        guard r != 0, input.size > 0 else { return input }
        let name = "cbv2_quant_rotation_\(input.dtype)_d\(d)_r\(r)_i\(inverse ? 1 : 0)"
        let kernel = lock.withLock {
            if let kernel = rotations[name] { return kernel }
            let kernel = MLXFast.metalKernel(
                name: name, inputNames: ["input"], outputNames: ["output"],
                source: rotateBody, header: header, ensureRowContiguous: true)
            rotations[name] = kernel
            return kernel
        }
        return kernel([input], template: [("T", input.dtype), ("D", d), ("R", r), ("INVERSE", inverse)],
                      grid: (d, input.size / d, 1), threadGroup: (d, 1, 1),
                      outputShapes: [input.shape], outputDTypes: [input.dtype], stream: stream)[0]
    }
}
