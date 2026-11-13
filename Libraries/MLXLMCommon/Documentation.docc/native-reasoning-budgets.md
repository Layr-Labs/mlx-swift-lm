# Native reasoning budgets

``EngineV2`` can bound one thought span with ``CBv2ReasoningBudget`` on a
``CBv2Request``. This control is off by default. Set
`DARKBLOOM_NATIVE_REASONING_BUDGET=1` before constructing the engine, or pass
`reasoningBudgetsEnabled: true` to its initializer. Check
``CBv2Engine/supportsReasoningBudgets`` before translating a client request.

The family adapter supplies a positive token limit, distinct atomic opening and
closing token IDs, and whether its rendered generation prompt already opened a
thought. Validate marker encode/decode round trips with the loaded tokenizer.
Do not infer this initial state from user text or historical messages.

```swift
request.reasoningBudget = try CBv2ReasoningBudget(
    maximumTokens: 512,
    openTokenID: validatedOpenToken,
    closeTokenID: validatedCloseToken,
    startsInsideThought: promptStartsInsideThought)
let events = try engine.submit(request)
```

Only confirmed payload tokens inside the thought spend the limit. A natural
close ends the thought early. After N payload tokens the next sample is the
closing marker, then normal answer sampling continues in the same request.
Repeated open markers are suppressed inside the thought and after its close;
there is no second quota. Output before the first open does not spend the
thought limit. A model that never opens a thought is unchanged.

The original total output limit, stop tokens and stop strings still apply, so
an output limit or stop can end the request before the forced close or answer.
The forced delimiter counts toward completion usage. Logprobs describe the raw
model distribution, including the forced delimiter. Cancellation and native
resource retirement use the existing engine lifecycle.

Budget rows use ordinary target decoding for their whole lifetime. They do not
use MTP or chained decode. A mixed batch can therefore lose decode chaining;
this interface makes no throughput claim. Requests without a budget keep their
existing sampling behavior. Membership changes rebuild budget state from
confirmed output, and retirement clears it before a request ID can be reused.

The default sampler supports this control. Other samplers and engines report
support only when they implement the complete lifecycle. ``EngineV2`` refuses
budgets when disabled, when the sampler does not support them, or when combined
with a generic token constraint. The native block engine refuses budgets.
Markers outside the model vocabulary fail before a token is emitted.
