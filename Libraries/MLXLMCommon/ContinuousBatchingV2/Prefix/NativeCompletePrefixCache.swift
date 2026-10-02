/// Optional native-profile store contract. A plain close() acknowledgement is
/// not proof that queued authenticated imports, exports and callbacks drained.
/// No default implementation: the real store must join its actual operations.
/// This capability is necessary, but does not itself grant a native work loan
/// or establish model/engine/store identity or successful GPU completion.
public protocol CBv2NativeCompletePrefixCache: CBv2CompletePrefixCache {
    func closeAndWait() async
}
