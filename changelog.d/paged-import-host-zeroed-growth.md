### Grow an imported prefix checkpoint's suffix backing from host-zeroed pages

- Restoring a complete prefix checkpoint into the paged KV pool grows backing for
  the request's remaining length. With `hostZeroedSegments`, admission growth
  wraps zero-fill-on-demand pages, but the restore created those segments with a
  device zero fill, an evaluator call and a stream fence. A restored request
  that asks for a long maximum length waited for that fill on the engine thread
  before its first prefill launch.
- The restore now grows its suffix backing the same way admission does. Pools
  without host zeros keep the device fill. Contents and accounting are unchanged.
