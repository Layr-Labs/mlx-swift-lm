#!/usr/bin/env python3
"""CPU reference of the Inkling 2 DSpark drafter (`mtp.safetensors`).

Written from the vLLM (PR 69028, `models/inkling2/nvidia/dspark.py`) and sglang
(PR 51926, `models/inkling2_mtp.py`) descriptions; it imports neither. fp32 math
with the bf16 roundings those kernels perform. Used to make the Swift fixtures:

    python inkling2_dspark_reference.py tiny OUT.safetensors
    python inkling2_dspark_reference.py real --mtp mtp.safetensors \
        [--target model.safetensors] [--features FEATURES.safetensors] OUT.safetensors

`tiny` builds a small random drafter and stores its weights with the expected
outputs. `real` uses the checkpoint drafter; without `--target` it uses a seeded
synthetic unembed/embedding, and without `--features` seeded synthetic target
features.
"""

import argparse
import json
import math
import struct
import sys

import torch
from safetensors.torch import load_file, save_file

BF16 = torch.bfloat16
F32 = torch.float32


def rms(x, w, eps):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w.float()


def rmsnorm(x, w, eps):
    return rms(x, w, eps).to(BF16)


def linear(x, w):
    return (x.float() @ w.float().T).to(BF16)


def mix_rmsnorm(features, alpha, w, eps):
    """features [T, taps, H] bf16, alpha fp32 [taps]."""
    mixed = torch.zeros(features.shape[0], features.shape[2], dtype=F32)
    for tap in range(features.shape[1]):
        mixed = mixed + features[:, tap].float() * alpha[tap]
    return rms(mixed, w, eps).to(BF16)


def rope_tables(positions, head_dim, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=F32) / head_dim))
    freqs = positions.to(F32)[:, None] * inv_freq[None, :]
    return freqs.cos(), freqs.sin()


def rope(x, positions, head_dim, theta):
    """neox RoPE of x [T, heads, D] (bf16 in, fp32 math, bf16 out)."""
    cos, sin = rope_tables(positions, head_dim, theta)
    cos, sin = cos[:, None, :], sin[:, None, :]
    half = head_dim // 2
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(BF16)


def head_norm(x, w, eps):
    """Per-head RMSNorm of x [T, heads, D], rounded to bf16."""
    return rms(x, w, eps).to(BF16)


class Drafter:
    def __init__(self, cfg, weights, embed, embed_norm, unembed, softcap, vocab):
        self.cfg = cfg
        self.w = weights
        self.embed = embed
        self.embed_norm = embed_norm
        self.unembed = unembed
        self.softcap = softcap
        self.vocab = vocab
        self.eps = cfg["rms_norm_eps"]
        self.alpha = torch.softmax(weights["fuse_weight"].float(), dim=-1)
        self.markov_topk = 0
        ho_w = weights["markov_head.hidden_out.weight"].double()
        ho_b = weights["markov_head.hidden_out.bias"].double()
        fold_w, fold_b = [], []
        for start in range(0, unembed.shape[0], 4096):
            head = unembed[start : start + 4096].double()
            fold_w.append((head @ ho_w).to(BF16))
            fold_b.append((head @ ho_b).to(BF16))
        self.fold_w = torch.cat(fold_w)
        self.fold_b = torch.cat(fold_b)

    def layer(self, i, name):
        return self.w[f"layers.{i}.{name}"]

    def context_kv(self, i, features, positions):
        """Context K (post norm + RoPE) and V of draft layer i: [T, kv, D]."""
        c = self.cfg
        d, kvh = c["head_dim"], c["num_key_value_heads"]
        ctx = mix_rmsnorm(features, self.alpha[i], self.layer(i, "dflare_norm.weight"), self.eps)
        k = linear(ctx, self.layer(i, "self_attn.k_proj_ctx.weight")).view(-1, kvh, d)
        v = linear(ctx, self.layer(i, "self_attn.v_proj_ctx.weight")).view(-1, kvh, d)
        k = head_norm(k, self.layer(i, "self_attn.k_norm_ctx.weight"), self.eps)
        k = rope(k, positions, d, c["rope_theta"])
        return k, v

    def attention(self, i, h, block_pos, ctx_k, ctx_v):
        c = self.cfg
        d, qh, kvh = c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]
        q = linear(h, self.layer(i, "self_attn.q_proj.weight")).view(-1, qh, d)
        k = linear(h, self.layer(i, "self_attn.k_proj.weight")).view(-1, kvh, d)
        v = linear(h, self.layer(i, "self_attn.v_proj.weight")).view(-1, kvh, d)
        q = rope(head_norm(q, self.layer(i, "self_attn.q_norm.weight"), self.eps), block_pos, d, c["rope_theta"])
        k = rope(head_norm(k, self.layer(i, "self_attn.k_norm.weight"), self.eps), block_pos, d, c["rope_theta"])
        if i in c["local_layer_ids"]:
            window = c["sliding_window_size"]
            ctx_k, ctx_v = ctx_k[-window:], ctx_v[-window:]
        keys = torch.cat([ctx_k, k]).float().repeat_interleave(qh // kvh, 1)
        vals = torch.cat([ctx_v, v]).float().repeat_interleave(qh // kvh, 1)
        scores = torch.einsum("qhd,khd->hqk", q.float(), keys) * d**-0.5
        out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vals).to(BF16)
        return linear(out.reshape(out.shape[0], -1), self.layer(i, "self_attn.o_proj.weight"))

    def mlp(self, i, h):
        gate = linear(h, self.layer(i, "mlp.gate_proj.weight")).float()
        up = linear(h, self.layer(i, "mlp.up_proj.weight")).float()
        return linear((gate / (1.0 + torch.exp(-gate)) * up).to(BF16), self.layer(i, "mlp.down_proj.weight"))

    def add_norm(self, x, residual, w):
        s = residual.float() + x.float()
        return rms(s, w, self.eps).to(BF16), s.to(BF16)

    def block_hidden(self, anchor, features, trace=None, ctx_cache=None):
        """Draft hidden [block, H] for the block anchored after `features`.
        `ctx_cache[i]` may hold layer i's context K/V of these positions."""
        c = self.cfg
        n, block = features.shape[0], c["block_size"]
        ctx_pos = torch.arange(n)
        block_pos = torch.arange(n, n + block)
        x = self.w["mask_embed"].expand(block, -1).clone()
        x[0] = rmsnorm(self.embed[anchor].view(1, -1), self.embed_norm, self.eps)[0]
        residual = x
        h = rmsnorm(x, self.layer(0, "input_layernorm.weight"), self.eps)
        for i in range(c["num_hidden_layers"]):
            if i > 0:
                h, residual = self.add_norm(x, residual, self.layer(i, "input_layernorm.weight"))
            if ctx_cache is None:
                ctx_k, ctx_v = self.context_kv(i, features, ctx_pos)
            else:
                ctx_k, ctx_v = ctx_cache[i]
            if trace is not None:
                trace[f"ctx_k.{i}"], trace[f"ctx_v.{i}"] = ctx_k, ctx_v
            a = self.attention(i, h, block_pos, ctx_k, ctx_v)
            h, residual = self.add_norm(a, residual, self.layer(i, "post_attention_layernorm.weight"))
            x = self.mlp(i, h)
            if trace is not None:
                trace[f"layer_out.{i}"] = x
        out, _ = self.add_norm(x, residual, self.w["norm.weight"])
        return out

    def propose(self, anchor, features, trace=None, ctx_cache=None):
        """Greedy block proposal: returns (drafts [block-1], hidden [block, H])."""
        c = self.cfg
        rank = c["markov_rank"]
        hidden = self.block_hidden(anchor, features, trace, ctx_cache)
        rows = hidden[1:]
        base = linear(rows, self.unembed)
        f_rnn = mix_rmsnorm(features[-1:], self.alpha[-1], self.w["markov_head.dflare_norm.weight"], self.eps)
        rnn_w = self.w["markov_head.rnn_proj.weight"].float()
        rnn_b = self.w["markov_head.rnn_proj.bias"].float()
        inputs = torch.cat([rows, f_rnn.expand(rows.shape[0], -1)], -1).float()
        context = rnn_b + inputs @ rnn_w[:, 2 * rank :].T
        state = torch.zeros(rank, dtype=BF16)
        prev = anchor
        drafts, margins, logits_rows = [], [], []
        for i in range(rows.shape[0]):
            step_in = torch.cat([state, self.w["markov_head.markov_w1.weight"][prev]]).float()
            raw = context[i] + step_in @ rnn_w[:, : 2 * rank].T
            gate = torch.sigmoid(raw[:rank].to(BF16))
            cand = torch.tanh(raw[rank : 2 * rank].to(BF16))
            out = raw[2 * rank :].to(BF16)
            state = gate * state + (1.0 - gate) * cand
            latent = torch.tanh(out)
            if self.markov_topk:
                # Experiment: Markov bias only on the base logits' top-k.
                cand = torch.topk(base[i, : self.vocab].float(), self.markov_topk).indices
                bias_c = (self.fold_b[cand].float() + latent.float() @ self.fold_w[cand].float().T).to(BF16)
                sub = ((base[i, cand] + bias_c).float() / self.softcap).tanh().mul(self.softcap).to(BF16)
                logits = torch.full((self.vocab,), float("-inf"))
                logits[cand] = sub.float()
            else:
                bias = (self.fold_b.float() + latent.float() @ self.fold_w.float().T).to(BF16)
                logits = ((base[i] + bias).float() / self.softcap).tanh().mul(self.softcap).to(BF16)
                logits = logits[: self.vocab].float()
            top = torch.topk(logits, 2)
            d = int(logits.argmax())
            drafts.append(d)
            margins.append(float(top.values[0] - top.values[1]))
            logits_rows.append(logits)
            prev = d
        if trace is not None:
            trace["base"] = base
            trace["f_rnn"] = f_rnn
            trace["context"] = context
            trace["markov_state"] = state
            trace["margins"] = torch.tensor(margins)
            trace["logits"] = torch.stack(logits_rows)
        return drafts, hidden


def acceptance(drafter, tokens, features, start):
    """Teacher-forced greedy acceptance on a sequence whose tokens after
    `start` are the target's own greedy continuation. Anchor p sees context
    positions < p; drafts are compared with tokens[p+1 : p+block]."""
    c = drafter.cfg
    total = len(tokens)
    depth = c["block_size"] - 1
    cache = [drafter.context_kv(i, features, torch.arange(total)) for i in range(c["num_hidden_layers"])]
    accepted = {}
    for p in range(start, total - 1):
        drafts, _ = drafter.propose(
            tokens[p], features[:p], ctx_cache=[(k[:p], v[:p]) for k, v in cache])
        truth = tokens[p + 1 : p + 1 + depth]
        n = 0
        while n < len(truth) and drafts[n] == truth[n]:
            n += 1
        accepted[p] = (n, len(truth))
    full = [n for n, width in accepted.values() if width == depth]
    by_position = [sum(1 for n in full if n > i) for i in range(depth)]
    conditional = [
        by_position[i] / (by_position[i - 1] if i else len(full)) if (by_position[i - 1] if i else len(full)) else None
        for i in range(depth)
    ]
    # Engine walk at fixed depth 7: each round emits accepted + 1 tokens.
    p, rounds, emitted = start, 0, 0
    while p < total - 1:
        n, _ = accepted[p]
        rounds += 1
        emitted += n + 1
        p += n + 1
    return {
        "anchors": len(accepted),
        "full_window_anchors": len(full),
        "mean_accepted_full_window": sum(full) / len(full) if full else None,
        "accepted_at_least": by_position,
        "conditional_acceptance": conditional,
        "walk_rounds": rounds,
        "walk_tokens_per_round": emitted / rounds if rounds else None,
        "per_anchor": {str(k): v[0] for k, v in accepted.items()},
    }


def tiny_config():
    return {
        "hidden_size": 64,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "intermediate_size": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "local_layer_ids": [0, 2],
        "sliding_window_size": 16,
        "block_size": 8,
        "markov_rank": 8,
        "markov_head_type": "rnn_hidden",
        "target_layer_ids": [5, 23, 41, 48],
    }


def tiny_weights(cfg, vocab, g):
    h, r = cfg["hidden_size"], cfg["markov_rank"]
    qd = cfg["num_attention_heads"] * cfg["head_dim"]
    kvd = cfg["num_key_value_heads"] * cfg["head_dim"]
    inter = cfg["intermediate_size"]

    def n(*shape, scale):
        return (torch.randn(*shape, generator=g) * scale).to(BF16)

    def norm_w(d):
        return (0.6 + 0.8 * torch.rand(d, generator=g)).to(BF16)

    w = {
        "fuse_weight": n(cfg["num_hidden_layers"] + 1, len(cfg["target_layer_ids"]), scale=1.0),
        "mask_embed": n(h, scale=1.0),
        "norm.weight": norm_w(h),
        "markov_head.dflare_norm.weight": norm_w(h),
        "markov_head.hidden_out.weight": n(h, r, scale=0.3),
        "markov_head.hidden_out.bias": n(h, scale=0.1),
        "markov_head.markov_w1.weight": n(vocab, r, scale=1.0),
        "markov_head.rnn_proj.weight": n(3 * r, 2 * r + 2 * h, scale=0.15),
        "markov_head.rnn_proj.bias": n(3 * r, scale=0.1),
    }
    for i in range(cfg["num_hidden_layers"]):
        p = f"layers.{i}."
        w[p + "dflare_norm.weight"] = norm_w(h)
        w[p + "input_layernorm.weight"] = norm_w(h)
        w[p + "post_attention_layernorm.weight"] = norm_w(h)
        w[p + "self_attn.q_proj.weight"] = n(qd, h, scale=h**-0.5)
        w[p + "self_attn.k_proj.weight"] = n(kvd, h, scale=h**-0.5)
        w[p + "self_attn.v_proj.weight"] = n(kvd, h, scale=h**-0.5)
        w[p + "self_attn.k_proj_ctx.weight"] = n(kvd, h, scale=h**-0.5)
        w[p + "self_attn.v_proj_ctx.weight"] = n(kvd, h, scale=h**-0.5)
        w[p + "self_attn.o_proj.weight"] = n(h, qd, scale=qd**-0.5)
        w[p + "self_attn.q_norm.weight"] = norm_w(cfg["head_dim"])
        w[p + "self_attn.k_norm.weight"] = norm_w(cfg["head_dim"])
        w[p + "self_attn.k_norm_ctx.weight"] = norm_w(cfg["head_dim"])
        w[p + "mlp.gate_proj.weight"] = n(inter, h, scale=h**-0.5)
        w[p + "mlp.up_proj.weight"] = n(inter, h, scale=h**-0.5)
        w[p + "mlp.down_proj.weight"] = n(h, inter, scale=inter**-0.5)
    return w


def synthetic_features(n, hidden, g, scales=(4.0, 8.0, 8.0, 1.0)):
    return torch.stack(
        [(torch.randn(n, hidden, generator=g) * s).to(BF16) for s in scales], dim=1
    )


def read_header(path):
    with open(path, "rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(size)), 8 + size


def read_tensor(path, name):
    """One bf16 tensor from a safetensors file without loading the rest."""
    header, base = read_header(path)
    meta = header[name]
    assert meta["dtype"] == "BF16", meta
    start, end = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + start)
        buf = bytearray(f.read(end - start))
    return torch.frombuffer(buf, dtype=BF16).view(meta["shape"]).clone()


def cases_for(drafter, features_list, anchors, prefix, out, trace_all=False):
    for c, (features, anchor) in enumerate(zip(features_list, anchors)):
        trace = {} if (trace_all or c == 0) else None
        drafts, hidden = drafter.propose(anchor, features, trace)
        out[f"{prefix}{c}.features"] = features.reshape(features.shape[0], -1)
        out[f"{prefix}{c}.anchor"] = torch.tensor([anchor], dtype=torch.int32)
        out[f"{prefix}{c}.drafts"] = torch.tensor(drafts, dtype=torch.int32)
        out[f"{prefix}{c}.hidden"] = hidden.clone()
        if trace is not None:
            for k, v in trace.items():
                out[f"{prefix}{c}.trace.{k}"] = v.contiguous().clone()
        margins = [round(float(m), 3) for m in trace["margins"]] if trace is not None else None
        print(f"case {c}: context={features.shape[0]} anchor={anchor} drafts={drafts} margins={margins}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tiny", "real", "acceptance", "oracle"])
    ap.add_argument("--oracle-dir")
    ap.add_argument("--markov-topk", type=int, default=0)
    ap.add_argument("--only")
    ap.add_argument("out")
    ap.add_argument("--mtp")
    ap.add_argument("--config")
    ap.add_argument("--target")
    ap.add_argument("--features")
    ap.add_argument("--seed", type=int, default=20261120)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    out = {}
    if args.mode == "oracle":
        import os

        cfg_all = json.load(open(args.config))
        cfg, text = cfg_all["mtp_config"], cfg_all["text_config"]
        drafter = Drafter(
            cfg, load_file(args.mtp), read_tensor(args.target, "model.llm.embed.weight"),
            read_tensor(args.target, "model.llm.embed_norm.weight"),
            read_tensor(args.target, "model.llm.unembed.weight"),
            text["final_logit_softcapping"], text["unpadded_vocab_size"])
        drafter.markov_topk = args.markov_topk
        depth = cfg["block_size"] - 1
        out, report, index = {}, {}, 0
        for name in sorted(os.listdir(args.oracle_dir)):
            if args.only and name not in args.only.split(","):
                continue
            pdir = os.path.join(args.oracle_dir, name)
            hidden_path = os.path.join(pdir, "hidden.safetensors")
            tf_path = os.path.join(pdir, "teacher_forced.safetensors")
            if not (os.path.exists(hidden_path) and os.path.exists(tf_path)):
                continue
            h = load_file(hidden_path)
            tf = load_file(tf_path)
            features = torch.stack(
                [h[f"dspark_feature_t{t}"].to(BF16) for t in cfg["target_layer_ids"]], dim=1)
            prompt = [int(t) for t in tf["input_ids"].tolist()]
            first = int(tf["argmax"][-1])
            greedy = None
            greedy_path = os.path.join(pdir, "greedy.json")
            if os.path.exists(greedy_path):
                greedy = json.load(open(greedy_path))["generated_ids"]
                assert greedy[0] == first, (name, greedy[0], first)
            trace = {}
            drafts, hidden = drafter.propose(first, features, trace)
            entry = {"prompt_tokens": len(prompt), "anchor": first, "drafts": drafts,
                     "margins": [round(float(m), 4) for m in trace["margins"]]}
            if greedy is not None:
                truth = greedy[1 : 1 + depth]
                n = 0
                while n < len(truth) and drafts[n] == truth[n]:
                    n += 1
                entry.update({"greedy_next": truth, "first_round_accepted": n})
            # Teacher-forced on the prompt itself (user/template text, a proxy).
            entry["prompt_teacher_forced"] = acceptance(drafter, prompt, features, 1)
            entry["prompt_teacher_forced"].pop("per_anchor")
            report[name] = entry
            out[f"case{index}.features"] = features.reshape(features.shape[0], -1).contiguous()
            out[f"case{index}.anchor"] = torch.tensor([first], dtype=torch.int32)
            out[f"case{index}.drafts"] = torch.tensor(drafts, dtype=torch.int32)
            out[f"case{index}.hidden"] = hidden.clone()
            out[f"case{index}.trace.margins"] = trace["margins"].clone()
            print(f"{name}: L={len(prompt)} anchor={first} drafts={drafts} "
                  f"greedy={entry.get('greedy_next')} accepted={entry.get('first_round_accepted')} "
                  f"prompt_tf_tokens_per_round={entry['prompt_teacher_forced']['walk_tokens_per_round']}",
                  file=sys.stderr)
            index += 1
        save_file(out, args.out, metadata={"source": "oracle hidden.safetensors (prompt contexts)", "cases": json.dumps(sorted(report))})
        with open(args.out + ".json", "w") as fh:
            json.dump(report, fh, indent=1)
        return
    if args.mode == "acceptance":
        cfg_all = json.load(open(args.config))
        cfg, text = cfg_all["mtp_config"], cfg_all["text_config"]
        w = load_file(args.mtp)
        drafter = Drafter(
            cfg, w, read_tensor(args.target, "model.llm.embed.weight"),
            read_tensor(args.target, "model.llm.embed_norm.weight"),
            read_tensor(args.target, "model.llm.unembed.weight"),
            text["final_logit_softcapping"], text["unpadded_vocab_size"])
        f = load_file(args.features)
        report = {}
        for name in sorted({k.rsplit(".", 1)[0] for k in f if k.endswith(".tokens")}):
            tokens = [int(t) for t in f[f"{name}.tokens"].tolist()]
            features = f[f"{name}.features"].to(BF16)
            features = features.view(features.shape[0], len(cfg["target_layer_ids"]), -1)
            start = int(f[f"{name}.prompt_length"].item())
            report[name] = acceptance(drafter, tokens, features, start)
            r = report[name]
            print(f"{name}: anchors={r['anchors']} mean_accepted={r['mean_accepted_full_window']} "
                  f"tokens_per_round={r['walk_tokens_per_round']} conditional={r['conditional_acceptance']}",
                  file=sys.stderr)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
        return
    if args.mode == "tiny":
        cfg = tiny_config()
        vocab_padded, vocab = 300, 290
        w = tiny_weights(cfg, vocab_padded, g)
        embed = (torch.randn(vocab_padded, cfg["hidden_size"], generator=g) * 2.0).to(BF16)
        embed_norm = (0.6 + 0.8 * torch.rand(cfg["hidden_size"], generator=g)).to(BF16)
        unembed = (torch.randn(vocab_padded, cfg["hidden_size"], generator=g) * 0.3).to(BF16)
        drafter = Drafter(cfg, w, embed, embed_norm, unembed, 30.0, vocab)
        out.update({f"weights.{k}": v.contiguous() for k, v in w.items()})
        out["target.embed"] = embed
        out["target.embed_norm"] = embed_norm
        out["target.unembed"] = unembed
        lengths = [1, 5, 16, 23, 40]
        feats = [synthetic_features(n, cfg["hidden_size"], g) for n in lengths]
        anchors = [int(torch.randint(0, vocab, (1,), generator=g)) for _ in lengths]
        cases_for(drafter, feats, anchors, "case", out, trace_all=True)
        out["fold_w"] = drafter.fold_w
        out["fold_b"] = drafter.fold_b
        meta = {"config": json.dumps(cfg), "vocab": str(vocab), "softcap": "30.0", "seed": str(args.seed)}
    else:
        cfg_all = json.load(open(args.config))
        cfg = cfg_all["mtp_config"]
        text = cfg_all["text_config"]
        vocab = text["unpadded_vocab_size"]
        w = load_file(args.mtp)
        if args.target:
            embed = read_tensor(args.target, "model.llm.embed.weight")
            embed_norm = read_tensor(args.target, "model.llm.embed_norm.weight")
            unembed = read_tensor(args.target, "model.llm.unembed.weight")
            source = "target"
        else:
            hidden, padded = cfg["hidden_size"], text["vocab_size"]
            embed = (torch.randn(padded, hidden, generator=g) * 0.02).to(BF16)
            embed_norm = torch.ones(hidden, dtype=BF16)
            unembed = (torch.randn(padded, hidden, generator=g) * 0.02).to(BF16)
            source = "synthetic"
        drafter = Drafter(cfg, w, embed, embed_norm, unembed, text["final_logit_softcapping"], vocab)
        if args.features:
            f = load_file(args.features)
            feats = [f[k].to(BF16).view(f[k].shape[0], 4, -1) for k in sorted(f) if k.endswith(".features")]
            anchors = [int(f[k][0]) for k in sorted(f) if k.endswith(".anchor")]
        else:
            lengths = [1, 7, 300, 1100]
            feats = [synthetic_features(n, cfg["hidden_size"], g) for n in lengths]
            anchors = [int(torch.randint(0, vocab, (1,), generator=g)) for _ in lengths]
        cases_for(drafter, feats, anchors, "case", out, trace_all=True)
        meta = {"config": json.dumps(cfg), "vocab": str(vocab), "target_source": source, "seed": str(args.seed)}
    save_file(out, args.out, metadata=meta)
    print(f"wrote {args.out} ({len(out)} tensors)", file=sys.stderr)


if __name__ == "__main__":
    main()
