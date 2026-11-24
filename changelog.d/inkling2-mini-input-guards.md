### Inkling 2 Mini: input and knob guards

- The processor refuses a request of more than 64 images before it decodes any
  of them, with `Inkling2ImageLimitError` ("Inkling 2 takes at most 64 images
  per request; this request has N"). 64 is the largest per-request image cap
  the Darkbloom provider gives a vision model (the Qwen3-VL family's). A server
  that caps images itself at 64 or fewer refuses first, so for it this is a
  backstop.
- The model-owned attention binds segment pages in place only when each page's
  local index fits the 24 bits of its table entry (`binding << 24 | local
  page`); a larger segment takes the gathered copy, which gives the same bits.
  Today's segments are far below 2^24 pages, so serving is unchanged.
- `DARKBLOOM_INKLING2_EXPERT_GROUP_ROWS` (8 to 64), `_FROM` (2 to 64),
  `_TILE_AT` (0 to 64) and `_UP_ROWS` (1 or 2) ignore a value outside their
  range and keep the default, as `_GROUP_BLOCK` does. Unset, they are unchanged.
