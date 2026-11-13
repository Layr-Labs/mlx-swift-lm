### Return an idle GLM pool's free KV segments

- The issued native GLM pool kept every segment it ever committed until it
  retired, and each complete-checkpoint restore attached more. Its free
  backing stayed charged as the slot's physical floor, so after long-prompt
  restores a lone request on an idle engine could not reserve its state and
  first step's scratch and failed `kv_capacity_exhausted`.
- The engine now returns the pool's whole free segments, and refunds their
  floor, before retrying a lone row whose step scratch was refused while the
  pool held free segments, and once it has been idle 30 seconds. While another
  request's checkpoint publication or retirement keeps the pool busy, that row
  waits up to 10 seconds instead of failing; it is retried once after a
  release. `DARKBLOOM_NATIVE_SEGMENT_RELEASE` sets the idle seconds; any value
  that is not a positive number (`0`, `off`) keeps the former behavior. Other
  pools are unchanged.
