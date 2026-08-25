# EigenLabs Qwen3.6 35B-A3B Swift Optimization Report

## Realistic context matrix

This matrix supersedes the earlier 129-token prompt / 100-token output
microbenchmark as the headline result. Each cell uses a non-repeating CPython
3.12 repository context with the coding task at the end, followed by exactly
1,024 generated tokens. Values are medians of three order-alternated runs under
the canonical GPU lock.

| prefill context | stock TTFT | optimized TTFT | TTFT reduction | stock prefill | optimized prefill | stock AR decode | fast K2 decode | decode uplift | current exact candidate | parity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1,024 | 380.2 ms | **285.9 ms** | **24.8%** | 2693.3 tok/s | **3582.1 tok/s** | 137.1 tok/s | **196.4 tok/s** | **43.3%** | 100.8 tok/s | **FAIL at token 13** |
| 16,384 | 7579.8 ms | **4804.3 ms** | **36.6%** | 2161.5 tok/s | **3410.3 tok/s** | 120.5 tok/s | **163.9 tok/s** | **36.0%** | 91.0 tok/s | **FAIL at token 13** |
| 32,768 | 16996.3 ms | **11309.7 ms** | **33.5%** | 1927.9 tok/s | **2897.3 tok/s** | 113.7 tok/s | **144.9 tok/s** | **27.4%** | 82.7 tok/s | **FAIL at token 0** |
| 65,536 | 40033.9 ms | **29025.4 ms** | **27.5%** | 1637.0 tok/s | **2257.9 tok/s** | 95.2 tok/s | **105.4 tok/s** | **10.7%** | 70.3 tok/s | **FAIL at token 62** |

Decode throughput times the 1,023 inter-token intervals after the first token.
The optimized prefill route is the 8,192-token construction-time chunk and solo
stripe. The fast decode route is fixed K2 with rectangular target-authoritative
verification.

## Byte-exact finding

The previous 100-token check was too short. Over the full 1,024-token workload,
the current serial `--output-parity byte-exact` route is deterministic but does
not remain byte-identical to stock. It diverges at generated token 13, 13, 0,
and 62 for the 1K, 16K, 32K, and 64K prompts and is slower than stock at every
context. It is rejected evidence, not a passing exact mode or a retained
optimization. The exact implementation must be replaced before the flag can be
presented as useful.

## Evidence

All 36 measured cells have both a machine-readable JSON receipt and a
human-readable Markdown report in [`context-matrix`](context-matrix). Every
receipt contains:

- all exact prompt and generated token IDs;
- submitted, first-token, and last-token timestamps;
- the requested and actual 1,024-token decode length;
- construction routes and MTP acceptance totals;
- model, source, and mlx-swift revisions;
- GPU-lock ownership and peak memory.

The issue/PR body contains the complete 36-row table with direct links to each
receipt. Route order changes by repetition:

1. stock, fast, current exact candidate;
2. current exact candidate, fast, stock;
3. fast, stock, current exact candidate.

## Exact pins

| item | value |
|---|---|
| model | `EigenLabs/Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8` |
| model revision | `73a03825c2226177f3e679210965dba3508cdee8` |
| mlx-swift-lm base | `ab73a827c9dde6f8802507003aa0be71605aab8e` |
| mlx-swift dependency | `606d28cfa8c1d66b2975d3162a4aac9756835c5f` |
| measured benchmark source | `aa5c23d` |
| chip | Apple M5 Max, 128 GiB unified memory |
| GPU lock | `/tmp/mtplx-gpu-exclusive.lock` |

The prompt fixtures tokenize to exactly 1,024, 16,384, 32,768, and 65,536
tokens under the model chat template. They are generated deterministically from
real CPython 3.12 standard-library source by
[`scripts/build-qwen36-python-contexts.py`](../../scripts/build-qwen36-python-contexts.py).

## Construction boundary

Construction inspection fixes the target contract at H2048, E256/top8, I512,
40 layers, affine W4/g64 target experts, W8/g64 target routers/shared gates,
and MXFP8/g32 inline-MTP matrices. Target affine readers are not transplanted
into the differently packed assistant.

The retained prefill and fast-decode routes are installed after the exact model
contract is validated. Enabled hot paths route only on values that actually
vary at runtime, such as logical M. They do not re-read model metadata or the
environment, increment proof counters, or silently fall back to stock.

## Current disposition

Retained:

- 8,192-token prefill chunk and solo stripe;
- fixed-K2 rectangular target-authoritative fast decode;
- exact-artifact row-owned E256/top8 router and M1/M2 expert reduction.

Rejected or still incomplete:

- the current serial byte-exact verifier: slower than stock and not exact over
  the full 1,024-token generation;
- a 16K prefill stripe: slower and higher-memory than the retained 8K stripe;
- fixed K1/K3/K4: slower than K2;
- transplanting target affine expert kernels into the MXFP8 assistant: invalid
  physical packing.

See [`OPTIMIZATION_LEDGER.md`](OPTIMIZATION_LEDGER.md) for the candidate history.
