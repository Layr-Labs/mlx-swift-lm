// Copyright © 2026 Eigen Labs.
// Retained final challenge27c821c4 rank and prefix-bound bodies.
enum Gemma4B8RouteSources {
    static let raw = """
            const uint assignment = thread_position_in_grid.x;
            const uint lane = thread_index_in_simdgroup;
            const uint key = (uint)indices[assignment];
            const uint key_low = (uint)indices[lane];
            const uint key_high = (uint)indices[32u + lane];
            uint rank = 0;
            #pragma clang loop unroll(full)
            for (uint source = 0; source < 32; ++source) {
                const uint other_low = simd_broadcast(key_low, ushort(source));
                rank += (other_low < key)
                    || (other_low == key && source < assignment);
                const uint other_high = simd_broadcast(key_high, ushort(source));
                const uint high_assignment = 32u + source;
                rank += (other_high < key)
                    || (other_high == key && high_assignment < assignment);
            }
            row_order[rank] = assignment >> 3;
            sorted_keys[rank] = key;
            inverse_order[assignment] = rank;
        """
    static let prefix = """
            const uint assignment = thread_position_in_grid.x;
            const uint lane = thread_index_in_simdgroup;
            const uint key = (uint)indices[assignment];
            const uint key_low = (uint)indices[lane];
            const uint key_high = (uint)indices[32u + lane];
            uint rank = 0;
            uint run_offset = 0;
            uint run_length = 0;
            #pragma clang loop unroll(full)
            for (uint source = 0; source < 32; ++source) {
                const uint other_low = simd_broadcast(key_low, ushort(source));
                rank += (other_low < key)
                    || (other_low == key && source < assignment);
                run_offset += other_low == key && source < assignment;
                run_length += other_low == key;
                const uint other_high = simd_broadcast(key_high, ushort(source));
                const uint high_assignment = 32u + source;
                rank += (other_high < key)
                    || (other_high == key && high_assignment < assignment);
                run_offset += other_high == key && high_assignment < assignment;
                run_length += other_high == key;
            }
            const uint run_remaining = run_length - run_offset;
            row_order[rank] = assignment >> 3;
            sorted_keys[rank] = 0x80000000u | key
                | (run_offset << 8) | ((run_remaining - 1) << 14);
            inverse_order[assignment] = rank;
        """
}
