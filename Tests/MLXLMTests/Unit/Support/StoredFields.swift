import Foundation

extension UnitTests {
    /// Lists the stored properties of a value as `path=value` lines, sorted by
    /// path.
    ///
    /// The lines include private properties, because `Mirror` reads every
    /// stored property. Dictionary and set elements are sorted, so the result
    /// does not depend on hash order. Tests compare these lines with the
    /// expected values of a decoded configuration.
    static func storedFields(of value: Any) -> [String] {
        var lines: [String] = []
        appendFields(of: value, path: "", to: &lines)
        return lines.sorted()
    }

    private static func appendFields(of value: Any, path: String, to lines: inout [String]) {
        let mirror = Mirror(reflecting: value)
        switch mirror.displayStyle {
        case .optional:
            if let wrapped = mirror.children.first?.value {
                appendFields(of: wrapped, path: path, to: &lines)
            } else {
                lines.append("\(path)=nil")
            }
        case .struct, .class:
            if mirror.children.isEmpty {
                lines.append("\(path)=\(String(describing: value))")
            }
            for child in mirror.children {
                guard let label = child.label, !label.hasPrefix("$__lazy_storage_$_") else {
                    continue
                }
                appendFields(
                    of: child.value, path: path.isEmpty ? label : "\(path).\(label)", to: &lines)
            }
        case .dictionary:
            let entries = mirror.children.map { entry -> (String, Any) in
                let pair = Mirror(reflecting: entry.value).children.map(\.value)
                return (String(describing: pair[0]), pair[1])
            }
            if entries.isEmpty {
                lines.append("\(path)=[:]")
            }
            for (key, element) in entries.sorted(by: { $0.0 < $1.0 }) {
                appendFields(of: element, path: "\(path)[\(key)]", to: &lines)
            }
        case .set:
            let elements = mirror.children.map { String(describing: $0.value) }.sorted()
            lines.append("\(path)=Set(\(elements.joined(separator: ", ")))")
        case .collection:
            if mirror.children.isEmpty {
                lines.append("\(path)=[]")
            }
            for (index, element) in mirror.children.enumerated() {
                appendFields(of: element.value, path: "\(path)[\(index)]", to: &lines)
            }
        default:
            lines.append("\(path)=\(String(describing: value))")
        }
    }
}
