// Copyright © 2026 Eigen Labs.
// Final challenge27c821c4 Gemma4PromptGlueV1 arithmetic, without KV packing
// or process-global diagnostics. Scalar twin changes only memory accesses.
enum Gemma4PromptGlueSources {
    static let vector = """
            typedef vec<T, 4> T4;
            constexpr uint cols4 = N / 4;
            const uint tid = thread_position_in_grid.x;
            const uint row = tid / cols4;
            if (row >= ROWS) return;
            const uint c4 = tid - row * cols4;
            const size_t base = size_t(row) * PITCH + size_t(c4) * 4;
            const T4 g4 = *reinterpret_cast<const device T4*>(gate + base);
            const T4 u4 = *reinterpret_cast<const device T4*>(up + base + UP_OFF);

            // The compiled kernel's constants: printed at seven significant
            // digits, cast to float, then to T (the `1` is an int32 cast).
            const T c_half = static_cast<T>(static_cast<float>(0.5));
            const T c_one = static_cast<T>(static_cast<int32_t>(1));
            const T c_k = static_cast<T>(static_cast<float>(0.7978846));
            const T c_a = static_cast<T>(static_cast<float>(0.044715));

            T4 o4;
            for (uint i = 0; i < 4; ++i) {
                const T g = g4[i];
                const T u = u4[i];
                const T hg = c_half * g;
                T t = c_a * g;
                t = t * g;
                t = t * g;
                t = g + t;
                t = c_k * t;
                t = metal::precise::tanh(t);
                t = c_one + t;
                t = hg * t;
                o4[i] = t * u;
            }
            *reinterpret_cast<device T4*>(out + size_t(row) * N + size_t(c4) * 4) = o4;
            """

    static let scalar = """
            typedef T T4[4];
            constexpr uint cols4 = N / 4;
            const uint tid = thread_position_in_grid.x;
            const uint row = tid / cols4;
            if (row >= ROWS) return;
            const uint c4 = tid - row * cols4;
            const size_t base = size_t(row) * PITCH + size_t(c4) * 4;
            T4 g4;
            T4 u4;
            for (uint i = 0; i < 4; ++i) {
                g4[i] = gate[base + i];
                u4[i] = up[base + UP_OFF + i];
            }

            // The compiled kernel's constants: printed at seven significant
            // digits, cast to float, then to T (the `1` is an int32 cast).
            const T c_half = static_cast<T>(static_cast<float>(0.5));
            const T c_one = static_cast<T>(static_cast<int32_t>(1));
            const T c_k = static_cast<T>(static_cast<float>(0.7978846));
            const T c_a = static_cast<T>(static_cast<float>(0.044715));

            T4 o4;
            for (uint i = 0; i < 4; ++i) {
                const T g = g4[i];
                const T u = u4[i];
                const T hg = c_half * g;
                T t = c_a * g;
                t = t * g;
                t = t * g;
                t = g + t;
                t = c_k * t;
                t = metal::precise::tanh(t);
                t = c_one + t;
                t = hg * t;
                o4[i] = t * u;
            }
            for (uint i = 0; i < 4; ++i) {
                out[size_t(row) * N + size_t(c4) * 4 + i] = o4[i];
            }
            """
}
