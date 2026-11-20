### Keep a refused GLM prefix restore's assistant charge on its stage

- A GLM-5.3-Flash request submitted with a first-token deadline prepares its
  complete-prefix restore, including the MTP assistant state, before the
  restore is published. When publication was refused, releasing the request
  moved that assistant's charge, which the checkpoint stage still held, into
  the engine's memory ledger as new growth. With no process memory headroom
  left, the ledger refused it and the provider stopped with a Swift trap.
- Releasing such a request now leaves that charge on the stage, which refunds
  it once the restore's native work retires. Published restores and ordinary
  requests release as before.
