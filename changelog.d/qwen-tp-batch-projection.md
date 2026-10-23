### Qwen tensor batch projection interfaces

Allow a Qwen-local projection to group tensor collectives over independent
decode rows while preserving each row's arithmetic. Mixed text/media rows keep
their original fused RoPE or explicit MRoPE operator. The native consumer owns
request histories and admission; these interfaces alone do not activate or
qualify clustered serving.
