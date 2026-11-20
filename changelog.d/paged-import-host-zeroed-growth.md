### Grow an imported prefix checkpoint's suffix backing from host-zeroed pages

- Restoring a complete prefix checkpoint into a (non-native) segmented paged KV
  pool grows backing for the request's remaining length. With
  `hostZeroedSegments`, admission growth wraps zero-fill-on-demand pages for
  eligible (VM-page-granular) new segments, but the restore created every new
  segment with a device zero fill, an evaluator call and a stream fence. A
  restored request that asks for a long maximum length waited for that fill on
  the engine thread before its first prefill launch.
- The restore now grows its suffix backing the same way admission does:
  eligible new segments wrap host pages, others keep the device fill. Pools
  without host zeros and native-bound restores are unchanged. Contents and the
  accounting rules are unchanged.
