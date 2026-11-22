### Inkling 2 Mini: reset the MTP depth policy's cost learning after a warm-up

- `EngineV2.resetMTPCostLearning()` clears the MTP depth policy's measured
  step costs (the raw per-depth estimates and the depth controller's
  learning) while the engine is idle; the next request measures them again.
  The policy keeps one plain-step cost sample per engine for its whole life,
  and a load warm-up's first steps run cold: on an M5 Ultra the warm-up
  recorded 96-99 ms for a one-row plain step that normally takes 15 ms (and
  245-298 ms at 2 and 8 rows), so every round looked free and the policy
  drafted at its maximum depth. A provider calls it after its warm-up.
  Served one-stream code on Inkling 2 Mini: 107 tok/s before, 121-123 after;
  tokens are unaffected. Test: `Inkling2DSparkEngineTests.costLearningResetKeepsTokens`.
