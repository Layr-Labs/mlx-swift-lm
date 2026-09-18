// Copyright © 2026 Eigen Labs.
// Final challenge27c821c4 affine-Q4/g64 lookup/dequantize/scale body.
// Valid token IDs retain the original gather domain, including negative wrap.
enum Gemma4ScaledEmbeddingSources {
    static let source = """
            const uint col = thread_position_in_grid.x;
            const uint row = thread_position_in_grid.y;
            // The launch is exactly one thread per packed word of one token
            // row, so the grid carries the row geometry with no shape buffer.
            const uint words_per_row = threads_per_grid.x;
            const uint groups_per_row = words_per_row >> 3;

            // Stock `weight[x]` gathers through `offset_neg_idx`: a negative
            // id wraps by the axis size. Positive out-of-range ids are
            // undefined in the stock gather too and are not redefined here.
            const int raw_token = tokens[row];
            const int vocab = w_shape[0];
            const size_t t = size_t(raw_token < 0 ? raw_token + vocab : raw_token);

            const uint packed = w[t * size_t(words_per_row) + size_t(col)];
            const size_t gindex = t * size_t(groups_per_row) + size_t(col >> 3);

            T scale = scales[gindex];
            T bias = biases[gindex];
            T es = embed_scale;

            device T* o = out
                + (size_t(row) * size_t(words_per_row) + size_t(col)) * 8;

            #pragma clang loop unroll(full)
            for (int i = 0; i < 8; i++) {
                uint8_t d = (packed >> (4 * i)) & 0x0f;
                // Boundary 1 — identical to `affine_dequantize`'s store.
                const T dequantized = scale * d + bias;
                // Boundary 2 — identical to the stock `* embedScale` multiply.
                o[i] = dequantized * es;
            }
            """
}
