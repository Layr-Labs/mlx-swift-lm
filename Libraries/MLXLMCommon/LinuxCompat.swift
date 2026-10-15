// Copyright © 2026 Eigen Labs.
//
// Minimal shims for APIs absent from swift-corelibs-foundation on Linux.

import Foundation
import MLX

#if !canImport(os)

    /// Drop-in for `os.Logger` where only `info` is used in this package.
    /// Messages are dropped. `LinuxLogMessage` accepts the same
    /// `\(value, privacy: .public)` interpolation as `OSLogMessage`, so the call
    /// sites are the same on every platform.
    public struct LinuxLogger: Sendable {
        public init(subsystem: String, category: String) {}

        public func info(_ message: LinuxLogMessage) {}
    }

    /// Stand-in for `OSLogPrivacy`.
    public enum LinuxLogPrivacy: Sendable {
        case `public`
        case `private`
    }

    /// Stand-in for `OSLogMessage`. It keeps no text.
    public struct LinuxLogMessage: ExpressibleByStringInterpolation, Sendable {
        public struct StringInterpolation: StringInterpolationProtocol {
            public init(literalCapacity: Int, interpolationCount: Int) {}
            public mutating func appendLiteral(_ literal: String) {}
            public mutating func appendInterpolation<T>(
                _ value: T, privacy: LinuxLogPrivacy = .private
            ) {}
        }

        public init(stringLiteral value: String) {}
        public init(stringInterpolation: StringInterpolation) {}
    }

    /// Drop-in for `OSSignposter` (signposts disabled on Linux).
    public struct LinuxSignposter: Sendable {
        public init(subsystem: String, category: String) {}

        public var isEnabled: Bool { false }

        public func beginInterval(_ name: StaticString) -> LinuxSignpostIntervalState? { nil }

        public func endInterval(_ name: StaticString, _ state: LinuxSignpostIntervalState?) {}
    }

    public struct LinuxSignpostIntervalState: Sendable {}

#endif

#if os(Linux)
    import Dispatch

    @inline(__always)
    public func lmWallTimeNow() -> Double {
        Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000
    }

    public enum LMPlatformCompat {
        /// Conservative cap when `GPU.deviceInfo()` is unavailable (CUDA Linux).
        public static let defaultMaxBufferSize = 1_073_741_824

        /// Distinguish JSON booleans from integer 0/1 without CoreFoundation.
        public static func nsNumberIsBoolean(_ number: NSNumber) -> Bool {
            String(cString: number.objCType) == "c"
        }

        public static func jsonQuantizationEntryIsFalse(_ raw: Any) -> Bool {
            if (raw as? Bool) == false { return true }
            if let number = raw as? NSNumber, nsNumberIsBoolean(number), !number.boolValue {
                return true
            }
            return false
        }

        public static func jsonQuantizationEntryIsTrueBool(_ raw: Any) -> Bool {
            if (raw as? Bool) == true { return true }
            if let number = raw as? NSNumber, nsNumberIsBoolean(number), number.boolValue {
                return true
            }
            return false
        }
    }
#else
    import CoreFoundation

    @inline(__always)
    public func lmWallTimeNow() -> CFAbsoluteTime {
        CFAbsoluteTimeGetCurrent()
    }

    public enum LMPlatformCompat {
        public static var defaultMaxBufferSize: Int {
            MLX.GPU.deviceInfo().maxBufferSize
        }

        public static func nsNumberIsBoolean(_ number: NSNumber) -> Bool {
            CFGetTypeID(number) == CFBooleanGetTypeID()
        }

        public static func jsonQuantizationEntryIsFalse(_ raw: Any) -> Bool {
            CFGetTypeID(raw as CFTypeRef) == CFBooleanGetTypeID() && (raw as? Bool) == false
        }

        public static func jsonQuantizationEntryIsTrueBool(_ raw: Any) -> Bool {
            CFGetTypeID(raw as CFTypeRef) == CFBooleanGetTypeID() && (raw as? Bool) == true
        }
    }
#endif
