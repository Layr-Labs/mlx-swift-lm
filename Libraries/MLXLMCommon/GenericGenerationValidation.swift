import Foundation

/// Optional model boundary for the generic TokenIterator execution path.
/// Native engines carry their own request state and do not use this iterator.
public protocol GenericGenerationValidating {
    func validateGenericGeneration() throws
}

public enum GenericGenerationError: Error, LocalizedError, Equatable {
    case nativeCBv2Required(modelType: String)

    public var errorDescription: String? {
        switch self {
        case .nativeCBv2Required(let modelType):
            return "\(modelType) requires native CBv2 generation; the generic TokenIterator path does not implement its complete sparse-attention and recurrent-state contract."
        }
    }
}
