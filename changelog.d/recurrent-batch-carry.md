### Keep the Qwen 3.5 recurrent batch between decode steps, and free a paused request's batch

- Each recurrent layer of a decode step split its new state into one-row
  views and the next step joined them back into a batch: a copy of every
  request's state, per layer, per step (48 layers and 146.8 MiB per request
  on the 27B). When the next step's rows are exactly the previous step's rows
  in the same order, the layer now takes the batched state as it is; any
  other mix joins as before. The values are the same either way.
  `DARKBLOOM_RECURRENT_BATCH_CARRY=0` turns the carry off.
- A request paused because its client stopped reading kept one-row views of
  its last step's batched state, and a view keeps the whole batch allocated
  for the pause: on an M5 Ultra about 1 GiB at 8 streams for the 27B. On
  pause the engine now copies the request's own rows out once, so the batch
  can be freed. `DARKBLOOM_RECURRENT_REOWN_ON_PAUSE=0` turns this off.
- BenchCBv2's paged arm builds its pool as serving does, queues a whole cell,
  can hash each request's output, can play a slow client, and applies
  serving's MLX memory limits.
