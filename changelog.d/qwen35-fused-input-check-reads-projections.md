### Qwen 3.5-family decode checks its fused input projection more cheaply

- Each Gated DeltaNet decode step re-checks that its four input projections are still frozen by reading only those projections and the layer's own frozen keys, instead of walking and flattening the layer's whole module tree. Outputs and freezing behaviour are unchanged; host time per token drops (about 6 % of the host thread in a Qwen 3.8 27B tensor-parallel profile).
