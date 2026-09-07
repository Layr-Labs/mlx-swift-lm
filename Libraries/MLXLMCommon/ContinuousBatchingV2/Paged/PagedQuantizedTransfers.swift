import MLX

/// Threadgroup-local quantization and reconstruction. Each record owns one
/// complete head/token; no two writers share a packed byte or scale entry.
enum PagedQuantizedTransfers {
    static let writeBody = """
        const uint d = thread_position_in_threadgroup.x;
        const int h = int(threadgroup_position_in_grid.y);
        const int r = int(threadgroup_position_in_grid.z);
        const int token = records[r * 3];
        const int page = records[r * 3 + 1];
        const int slot = records[r * 3 + 2];
        constexpr int GROUPS = D / G;
        constexpr int KDATA = D * KB / 8;
        constexpr int VDATA = D * VB / 8;
        constexpr int KROW = KDATA + 8 * GROUPS;
        constexpr int VROW = VDATA + 8 * GROUPS;
        threadgroup float k[D];
        threadgroup float v[D];
        threadgroup float scales[2 * GROUPS];
        threadgroup float offsets[2 * GROUPS];
        const int64_t key_source = (int64_t)h * keys_strides[0] + (int64_t)token * keys_strides[1] + (int64_t)d * keys_strides[2];
        const int64_t value_source = (int64_t)h * values_strides[0] + (int64_t)token * values_strides[1] + (int64_t)d * values_strides[2];
        const float raw = float(keys[key_source]);
        k[d] = R == 0 ? raw : raw * cbv2::quant_sign(d % max(R, 1));
        v[d] = float(values[value_source]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1; stride < R; stride <<= 1) {
            const float a = k[d];
            const float b = k[d ^ stride];
            const float next = (d & stride) ? b - a : a + b;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            k[d] = next;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (R != 0) k[d] *= rsqrt(float(R));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (d < GROUPS) {
            float klo = k[d * G], khi = klo;
            float vlo = v[d * G], vhi = vlo;
            for (int i = 1; i < G; i++) {
                klo = min(klo, k[d * G + i]); khi = max(khi, k[d * G + i]);
                vlo = min(vlo, v[d * G + i]); vhi = max(vhi, v[d * G + i]);
            }
            // Dividing first avoids overflow for opposite FP32 extremes.
            scales[d] = khi / float((1 << KB) - 1) - klo / float((1 << KB) - 1);
            scales[GROUPS + d] = vhi / float((1 << VB) - 1) - vlo / float((1 << VB) - 1);
            offsets[d] = klo;
            offsets[GROUPS + d] = vlo;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (page > 0) {
            const size_t row = ((size_t)page * H + h) * S + slot;
            device uchar* kb = const_cast<device uchar*>(storage) + row * KROW;
            device uchar* vb = const_cast<device uchar*>(storage) + VBASE + row * VROW;
            if (d < GROUPS) {
                reinterpret_cast<device float*>(kb + KDATA)[d] = scales[d];
                reinterpret_cast<device float*>(kb + KDATA)[GROUPS + d] = offsets[d];
                reinterpret_cast<device float*>(vb + VDATA)[d] = scales[GROUPS + d];
                reinterpret_cast<device float*>(vb + VDATA)[GROUPS + d] = offsets[GROUPS + d];
            }
            if (d < KDATA) {
                uint packed = 0;
                for (int i = 0; i < 8 / KB; i++) {
                    const int c = d * (8 / KB) + i;
                    const float scale = scales[c / G];
                    const float code = scale == 0.0f ? 0.0f
                        : round(k[c] / scale - offsets[c / G] / scale);
                    packed |= uint(clamp(code, 0.0f, float((1 << KB) - 1))) << (i * KB);
                }
                kb[d] = uchar(packed);
            }
            if (d < VDATA) {
                uint packed = 0;
                for (int i = 0; i < 8 / VB; i++) {
                    const int c = d * (8 / VB) + i;
                    const float scale = scales[GROUPS + c / G];
                    const float code = scale == 0.0f ? 0.0f
                        : round(v[c] / scale - offsets[GROUPS + c / G] / scale);
                    packed |= uint(clamp(code, 0.0f, float((1 << VB) - 1))) << (i * VB);
                }
                vb[d] = uchar(packed);
            }
        }
        if (h == 0 && d == 0 && r == 0) fence[0] = previous[0] + 1;
        """

    /// Gather returns native-basis K and V to preserve the sequence-cache
    /// snapshot/restore interface. Only prefill/debug paths materialize them.
    static let readBody = """
        const uint d = thread_position_in_threadgroup.x;
        const int h = int(threadgroup_position_in_grid.y);
        const int r = int(threadgroup_position_in_grid.z);
        const int token = records[r * 3];
        const int page = records[r * 3 + 1];
        const int slot = records[r * 3 + 2];
        const int count = output_shape[3];
        constexpr int KROW = D * KB / 8 + 8 * (D / G);
        constexpr int VROW = D * VB / 8 + 8 * (D / G);
        const size_t row = ((size_t)page * H + h) * S + slot;
        threadgroup float k[D];
        k[d] = cbv2::quant_load<KB, D, G>(storage + row * KROW, d);
        const float value = cbv2::quant_load<VB, D, G>(storage + VBASE + row * VROW, d);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1; stride < R; stride <<= 1) {
            const float a = k[d];
            const float b = k[d ^ stride];
            const float next = (d & stride) ? b - a : a + b;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            k[d] = next;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float key = R == 0 ? k[d] : k[d] * rsqrt(float(R)) * cbv2::quant_sign(d % max(R, 1));
        const size_t target = ((size_t)h * count + token) * D + d;
        device T* destination = const_cast<device T*>(output);
        destination[target] = T(key);
        destination[(size_t)H * count * D + target] = T(value);
        if (h == 0 && d == 0 && r == 0) fence[0] = previous[0] + 1;
        """
}
