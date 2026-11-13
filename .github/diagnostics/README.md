# Audio reader class control

This opt-in diagnostic compares the original strict audio tests and reader
matrix on one allocated macOS runner. It does not change audio runtime behavior
or replace the normal full-CI gates. The control source is frozen at
`154ceccf3b16ad45a19d99344c025f0c12a66ef0`; all 31 dependency pins must match the
saved graph. The workflow refuses hosted fallback or a different hardware class.

The `audio-reader-control` label triggers the separate workflow only on the
same-repository `astra/audio-reader-control-a` pull request. Apply it once after
the prior diagnostic run is complete and the control head is reviewed. No
allocation retry or capacity-setting change is part of this experiment.

Seven original tests run first, then two matrix tests in a separate process.
The runner requires nonempty binary/metallib identity, exact selected test IDs,
no skipped tests, and process-group drainage between phases. Timeout, cleanup
failure, ordinary test failure and identity refusal remain distinct outcomes.
Raw streams, identities and verdicts are uploaded even on failure. An ordinary
failure still permits matrix collection; failed cleanup prevents the next phase.

The historical failing build reported M5 Max Virtual, 12 CPUs, 32 GiB and
Xcode 27.2 (27B52). Its OS/framework identity was not recorded. Matching hardware
and pins alone cannot establish that every environmental variable matches.
Neither a pass nor a failure alone proves an audio cause or fix.

Run the non-audio runner regressions without Swift or GPU execution:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 .github/diagnostics/test_audio_reader_control.py -v
```

The regression's signal-denial case deliberately simulates failed cleanup,
then kills/reaps only its fixture-owned group. Python may emit a ResourceWarning
before that explicit fixture cleanup.
