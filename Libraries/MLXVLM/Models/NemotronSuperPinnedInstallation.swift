extension NemotronSuperInstallation.Installed {
    /// True when this installation was loaded from the pinned NVIDIA
    /// checkpoint (its index digest), the artifact whose full-model
    /// measurements, such as `NemotronHModel.prefillActivationBoundBytes`,
    /// apply. Synthetic fixtures are never pinned.
    public var isPinnedCheckpoint: Bool {
        receipt.indexSHA256 == NemotronSuperCheckpointPin.index
    }
}
