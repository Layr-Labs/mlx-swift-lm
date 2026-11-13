// Copyright © 2026 Eigen Labs.

import Foundation
import MLX

/// Reads arrays from an uncompressed `.npz` (what `numpy.savez` writes): a
/// ZIP of stored `.npy` members. Test input only: little-endian float32,
/// float64 and int32 arrays in C order.
struct MuseGlimmerDFlashNPZ {
    enum Failure: Error { case malformed(String), missing(String), unsupported(String) }

    private let data: Data
    private var members: [String: (offset: Int, size: Int)] = [:]

    init(_ url: URL) throws {
        data = try Data(contentsOf: url, options: .mappedIfSafe)
        let end = try endOfCentralDirectory()
        var cursor = end.offset
        for _ in 0 ..< end.entries {
            guard u32(cursor) == 0x0201_4b50 else { throw Failure.malformed("central header") }
            let method = u16(cursor + 10)
            var compressed = Int(u32(cursor + 20))
            var size = Int(u32(cursor + 24))
            let nameLength = Int(u16(cursor + 28))
            let extraLength = Int(u16(cursor + 30))
            let commentLength = Int(u16(cursor + 32))
            var local = Int(u32(cursor + 42))
            let name = String(
                decoding: data[(cursor + 46) ..< (cursor + 46 + nameLength)], as: UTF8.self)
            // ZIP64 extra: the 64-bit values of the fields that overflowed, in order.
            var extra = cursor + 46 + nameLength
            let extraEnd = extra + extraLength
            while extra + 4 <= extraEnd {
                let (id, length) = (u16(extra), Int(u16(extra + 2)))
                if id == 0x0001 {
                    var field = extra + 4
                    if size == 0xFFFF_FFFF {
                        size = Int(u64(field))
                        field += 8
                    }
                    if compressed == 0xFFFF_FFFF {
                        compressed = Int(u64(field))
                        field += 8
                    }
                    if local == 0xFFFF_FFFF { local = Int(u64(field)) }
                }
                extra += 4 + length
            }
            guard method == 0, compressed == size else {
                throw Failure.unsupported("compressed member \(name)")
            }
            guard u32(local) == 0x0403_4b50 else { throw Failure.malformed("local header") }
            let start = local + 30 + Int(u16(local + 26)) + Int(u16(local + 28))
            guard start + size <= data.count else { throw Failure.malformed(name) }
            members[name.hasSuffix(".npy") ? String(name.dropLast(4)) : name] = (start, size)
            cursor = extraEnd + commentLength
        }
    }

    var names: [String] { members.keys.sorted() }

    private func u16(_ offset: Int) -> UInt16 {
        UInt16(data[data.startIndex + offset]) | UInt16(data[data.startIndex + offset + 1]) << 8
    }
    private func u32(_ offset: Int) -> UInt32 {
        UInt32(u16(offset)) | UInt32(u16(offset + 2)) << 16
    }
    private func u64(_ offset: Int) -> UInt64 {
        UInt64(u32(offset)) | UInt64(u32(offset + 4)) << 32
    }

    private func endOfCentralDirectory() throws -> (offset: Int, entries: Int) {
        var position = data.count - 22
        while position >= max(0, data.count - 22 - 65_535) {
            if u32(position) == 0x0605_4b50 { break }
            position -= 1
        }
        guard position >= 0, u32(position) == 0x0605_4b50 else {
            throw Failure.malformed("end of central directory")
        }
        var entries = Int(u16(position + 10))
        var offset = Int(u32(position + 16))
        if entries == 0xFFFF || offset == 0xFFFF_FFFF {
            let locator = position - 20
            guard locator >= 0, u32(locator) == 0x0706_4b50 else {
                throw Failure.malformed("zip64 locator")
            }
            let record = Int(u64(locator + 8))
            guard u32(record) == 0x0606_4b50 else { throw Failure.malformed("zip64 record") }
            entries = Int(u64(record + 32))
            offset = Int(u64(record + 48))
        }
        return (offset, entries)
    }

    /// The member's payload bytes, its dtype string and its shape.
    private func member(_ name: String) throws -> (bytes: Data, descr: String, shape: [Int]) {
        guard let (start, size) = members[name] else { throw Failure.missing(name) }
        let base = data.startIndex + start
        guard size >= 10, data[base] == 0x93 else { throw Failure.malformed(name) }
        let major = data[base + 6]
        let headerLength = major == 1 ? Int(u16(start + 8)) : Int(u32(start + 8))
        let headerStart = start + (major == 1 ? 10 : 12)
        let header = String(
            decoding: data[
                (data.startIndex + headerStart) ..< (data.startIndex + headerStart + headerLength)],
            as: UTF8.self)
        func value(after key: String, until terminator: Character) throws -> String {
            guard let range = header.range(of: "'\(key)':") else { throw Failure.malformed(name) }
            let rest = header[range.upperBound...]
            guard let end = rest.firstIndex(of: terminator) else { throw Failure.malformed(name) }
            return String(rest[..<end])
        }
        let descr = try value(after: "descr", until: ",")
            .trimmingCharacters(in: CharacterSet(charactersIn: " '"))
        guard try value(after: "fortran_order", until: ",").contains("False") else {
            throw Failure.unsupported("fortran order \(name)")
        }
        let shapeText = try value(after: "shape", until: ")")
        let shape = shapeText.split(whereSeparator: { !$0.isNumber }).compactMap { Int($0) }
        let payload = headerStart + headerLength
        return (
            data[(data.startIndex + payload) ..< (data.startIndex + start + size)], descr, shape
        )
    }

    func array(_ name: String) throws -> MLXArray {
        let (bytes, descr, shape) = try member(name)
        let dtype: DType
        switch descr {
        case "<f4": dtype = .float32
        case "<f8": dtype = .float64
        case "<i4": dtype = .int32
        default: throw Failure.unsupported("\(name) \(descr)")
        }
        guard bytes.count == shape.reduce(dtype.size, *) else { throw Failure.malformed(name) }
        return MLXArray(Data(bytes), shape, dtype: dtype)
    }

    /// Rows `rows` of a float32 member whose first axis is the row axis.
    func rows(_ name: String, _ rows: Range<Int>) throws -> MLXArray {
        let (bytes, descr, shape) = try member(name)
        guard descr == "<f4", let count = shape.first, rows.lowerBound >= 0,
            rows.upperBound <= count, !rows.isEmpty
        else { throw Failure.unsupported("\(name) rows") }
        let width = shape.dropFirst().reduce(4, *)
        let start = bytes.startIndex + rows.lowerBound * width
        return MLXArray(
            Data(bytes[start ..< (start + rows.count * width)]),
            [rows.count] + Array(shape.dropFirst()), dtype: .float32)
    }

    func shape(_ name: String) throws -> [Int] { try member(name).shape }

    func ints(_ name: String) throws -> [Int] {
        try array(name).reshaped([-1]).asArray(Int32.self).map(Int.init)
    }

    func floats(_ name: String) throws -> [Float] {
        try array(name).asType(.float32).reshaped([-1]).asArray(Float.self)
    }
}
