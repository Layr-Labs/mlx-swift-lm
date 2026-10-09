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
        template<typename NATIVE, int D, int S, int G, int KB, int VB>
        struct PagedMixedQuantizedKVPage {
            const device uchar* key;
            const device uchar* value;
            const device NATIVE* native_keys;
            const device NATIVE* native_values;
            int native_start;
            int native_count;
            int logical_page;
            int64_t native_head_stride;
            int64_t native_token_stride;
            int64_t native_feature_stride;
            int64_t native_value_head_stride;
            int64_t native_value_token_stride;
            int64_t native_value_feature_stride;
            template<int EPT>
            void load(size_t base, uint lane, thread float* k, thread float* v) const {
                const size_t row = base / D;
                const int pos = logical_page * S + int(row % S);
                const bool recent = pos >= native_start && pos - native_start < native_count;
                constexpr int KROW = D * KB / 8 + 8 * (D / G);
                constexpr int VROW = D * VB / 8 + 8 * (D / G);
                for (int e = 0; e < EPT; e++) {
                    const int column = lane * EPT + e;
                    if (recent) {
                        const int64_t key_offset = int64_t(row / S) * native_head_stride
                            + int64_t(pos - native_start) * native_token_stride
                            + int64_t(column) * native_feature_stride;
                        const int64_t value_offset = int64_t(row / S) * native_value_head_stride
                            + int64_t(pos - native_start) * native_value_token_stride
                            + int64_t(column) * native_value_feature_stride;
                        k[e] = float(native_keys[key_offset]);
                        v[e] = float(native_values[value_offset]);
                    } else {
                        k[e] = quant_load<KB, D, G>(key + row * KROW, column);
                        v[e] = quant_load<VB, D, G>(value + row * VROW, column);
                    }
                }
            }
            template<int WIDTH, int TG, typename T>
            void write(size_t, const device T*, const device T*, int) const {}
        };
        template<typename NATIVE, int N, int D, int S, int G, int KB, int VB>
        struct PagedMixedQuantizedSegmentAccessor {
            const device uchar* buffers[N];
            const device int64_t* value_offsets;
            const device int32_t* record;
            int kv_heads;
            const device NATIVE* native_keys;
            const device NATIVE* native_values;
            int native_start;
            int native_count;
            int64_t native_head_stride;
            int64_t native_token_stride;
            int64_t native_feature_stride;
            int64_t native_value_head_stride;
            int64_t native_value_token_stride;
            int64_t native_value_feature_stride;
            using Page = PagedMixedQuantizedKVPage<NATIVE, D, S, G, KB, VB>;
            bool is_native_position(int pos) const {
                return pos >= native_start && pos - native_start < native_count;
            }
            Page page(int binding, int local, int logical) const {
                constexpr int KROW = D * KB / 8 + 8 * (D / G);
                constexpr int VROW = D * VB / 8 + 8 * (D / G);
                const size_t kp = (size_t)kv_heads * S * KROW;
                const size_t vp = (size_t)kv_heads * S * VROW;
                if (binding < 0 || binding >= N || local < 0
                    || (size_t)local >= (size_t)value_offsets[binding] / kp) {
                    binding = 0; local = 0;
                }
                return {buffers[binding] + (size_t)local * kp,
                    buffers[binding] + value_offsets[binding] + (size_t)local * vp,
                    native_keys, native_values, native_start, native_count, logical,
                    native_head_stride, native_token_stride, native_feature_stride,
                    native_value_head_stride, native_value_token_stride, native_value_feature_stride};
            }
            Page write_page(int logical) const { return page(record[4], record[5], logical); }
            Page read_page(const device int32_t*, int logical, int) const {
                const int index = logical - record[2];
                if (index < 0 || index >= record[3]) return page(0, 0, logical);
                return page(record[8 + 2 * index], record[9 + 2 * index], logical);
            }
        };
        }
        """

}
