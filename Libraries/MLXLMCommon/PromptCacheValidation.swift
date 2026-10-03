import MLX

/// Validate file input before calling nonthrowing cache setters. Empty state
/// means an unpopulated cache, not a missing keys/values pair.
internal func validatePromptCacheState(
    className: String, state: [MLXArray], metaState: [String]
) throws {
    func reject() throws {
        throw KVCacheError(message: "Invalid \(className) state or metadata")
    }
    func integers(_ count: Int) -> [Int]? {
        guard metaState.count == count else { return nil }
        let values = metaState.compactMap(Int.init)
        return values.count == count ? values : nil
    }
    switch className {
    case "KVCache", "KVCacheSimple", "ChunkedKVCache", "RotatingKVCache":
        guard state.isEmpty || (state.count == 2 && state.allSatisfy { $0.ndim == 4 }) else {
            try reject()
            return
        }
        if state.count == 2 {
            guard state[0].shape.prefix(3) == state[1].shape.prefix(3) else {
                try reject()
                return
            }
        }
        switch className {
        case "ChunkedKVCache":
            guard metaState.count == 2, let start = Int(metaState[1]), start >= 0,
                metaState[0] == "None" || Int(metaState[0]).map({ $0 >= 0 }) == true,
                start <= Int.max - (state.first?.dim(2) ?? 0)
            else {
                try reject()
                return
            }
        case "RotatingKVCache":
            guard let values = integers(5), values[0] >= 0, values[1] > values[0],
                values[2] > 0, values[3] >= 0, values[4] >= 0,
                values[4] <= (state.first?.dim(2) ?? 0),
                !state.isEmpty || values[3] == 0
            else {
                try reject()
                return
            }
        default:
            guard metaState.isEmpty || metaState == [""] else {
                try reject()
                return
            }
        }
    case "QuantizedKVCache":
        guard let values = integers(4), values[0] > 0, values[1] >= 0,
            values[2] > 0, values[3] > 0,
            state.isEmpty
                || ((state.count == 4 || state.count == 6)
                    && state.allSatisfy { $0.ndim == 4 && $0.dim(2) >= values[1] }),
            !state.isEmpty || values[1] == 0
        else {
            try reject()
            return
        }
    case "ArraysCache", "MambaCache":
        // Preserve the documented legacy compact-state format.
        if metaState.isEmpty || metaState == [""] { return }
        guard (2 ... 3).contains(metaState.count), let slots = Int(metaState[0]), slots >= 0
        else {
            try reject()
            return
        }
        let parts =
            metaState[1].isEmpty
            ? [] : metaState[1].split(separator: ",", omittingEmptySubsequences: false)
        let present = parts.compactMap { Int($0) }
        guard present.count == parts.count, present.count == state.count,
            Set(present).count == present.count, present.allSatisfy({ $0 >= 0 && $0 < slots })
        else {
            try reject()
            return
        }
        if metaState.count == 3 && !metaState[2].isEmpty {
            let padding = metaState[2].split(separator: ",", omittingEmptySubsequences: false)
            // advance() subtracts consumed tokens, so valid padding can be negative.
            guard padding.allSatisfy({ Int($0) != nil })
            else {
                try reject()
                return
            }
        }
    default:
        break  // CacheList validates its own flattened child inventory.
    }
}
