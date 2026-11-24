### Price the GLM step envelope for the tokens admission admits

- `EngineV2` checks, when it is built, that the worst GLM step fits the engine's
  admissible bytes and scratch limit. It priced that step for every row at the
  context limit, or the pool's whole token capacity when that is smaller. A
  caller whose admission reserves KV and step scratch for each request token
  can never hold that much history, so a smaller grant could run fewer rows
  than its admission allows.
- `EngineV2.init` takes an optional `glm5NextAdmittedBytesPerContextToken`, the
  charge the caller's admission reserves per request token, and keeps it as a
  property of the same name. With it, the check also bounds history by
  (admissible bytes − setup bytes) / max(charge, KV + step scratch per token).
  Nil, the default, keeps the previous check.
