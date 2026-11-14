### Fewer no-op GPU kernels per decode step

- A paged layer cache advances its per-row position offsets from the host
  mirror it already keeps in lockstep (`hostPositionOffsets`) instead of a GPU
  `+ length` per layer per step, and the segmented merge's unused sinks input
  (models without attention sinks) is built on the host instead of filled on
  the GPU. GLM 5 Next's offset advance now goes through the same path, so its
  host mirror also stays in lockstep.
- Gemma 4: decode expert gathers take host row indices, as Flint's do, instead
  of one GPU `Arange` per projection, and the attention's position snapshot no
  longer adds a `+ 0` copy kernel per layer.
- Same values, same output bits. On an M5 Ultra this removes about 180 small
  kernels (60 integer adds, 30 fills, 90 index ranges) from a one-stream Gemma 4
  26B-A4B decode step (about 1.3 % faster);
  the Qwen 3.8 27B step is unchanged in speed.
