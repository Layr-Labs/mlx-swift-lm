### Qwen media preparation and explicit position seams

Expose deterministic Qwen3.5-family media preprocessing/geometry and a
vision-only serial loading seam, plus request-owned embeddings and three-axis
positions for language forwards. Ordinary wrapper preprocessing uses the shared
implementation; default text arithmetic is unchanged. Tiny CPU coverage does
not imply full-model or distributed media qualification.
