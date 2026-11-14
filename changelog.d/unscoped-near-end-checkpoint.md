### Near-end checkpoints for local requests

- A request without a cache scope (a local caller's) now gets the
  end-of-prompt boundary (`demandedLatestCheckpointMinimumTokens`) and the
  capture-boundary partition (`demandedCaptureBoundaryMinimumTokens`), as a
  scoped request already does. Its checkpoints were already captured and
  restored under the cache-level salt, but the scheduler cut no range for it,
  so the deepest checkpoint of a solo prompt could sit up to a stripe below
  its end and the next turn re-prefilled that gap. Only where a range ends
  changes; reuse and capture keep their scope, identity and allocation gates.
  The short-checkpoint partition still requires a scope.
