### A GLM request whose prefix restore is refused prefills with its own assistant binding

- When a GLM-5.3-Flash prefix restore was refused, an ordinary request
  continued with a cold prefill. Until the refused restore's native work
  retired, the request's MTP assistant could be configured on the restore's
  binding, whose memory the checkpoint stage still held. If the restore then
  retired while the request was running, it found the request's allocations on
  that binding and the engine stopped with a native completion failure.
- The cold request now gets its own binding. The refused restore's binding is
  retired in place and keeps its charge on the stage until the restore's
  native work retires.
