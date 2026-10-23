### Ordered captured chat logprobs

- Carry captured token metadata through the same ordered server event stream as text, so a fast producer cannot attach future-token metadata to an earlier chat frame.
- Preserve absent metadata fields and collected response order. Native capture support and reasoning/tool source-span attribution are unchanged.
