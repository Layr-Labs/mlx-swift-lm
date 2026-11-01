# Adapted from Prince Canuma / mlx-vlm test_models.py at
# 1a6ebd06459be3b18cb1c7922305c9e362612c62 (MIT).
# See ThirdPartyNotices/Flint.md. No MLX import: independent NumPy float64 oracle.
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def reference(config, weights, ids):
    def linear(x, name):
        return x @ weights[name + ".weight"].T

    def norm(x, name):
        return (
            x
            / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + config.rms_norm_eps)
            * weights[name + ".weight"]
        )

    def mlp(x, prefix):
        gate = linear(x, prefix + ".gate_proj")
        return linear(
            gate / (1 + np.exp(-gate)) * linear(x, prefix + ".up_proj"),
            prefix + ".down_proj",
        )

    x = weights["model.embed_tokens.weight"][ids]
    batch, length, _ = x.shape
    positions = np.arange(length)
    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}"
        a = prefix + ".self_attn"
        hidden = norm(x, prefix + ".input_layernorm")
        q = norm(
            linear(hidden, a + ".q_proj").reshape(
                batch, length, config.num_attention_heads, config.head_dim
            ),
            a + ".q_norm",
        )
        k = norm(
            linear(hidden, a + ".k_proj").reshape(
                batch, length, config.num_key_value_heads, config.head_dim
            ),
            a + ".k_norm",
        )
        v = linear(hidden, a + ".v_proj").reshape(
            batch, length, config.num_key_value_heads, config.head_dim
        )
        if i not in config.no_rope_layer_indices:
            angles = positions[:, None] / config.rope_theta ** (
                np.arange(0, config.head_dim, 2) / config.head_dim
            )
            angles = np.concatenate([angles, angles], -1)[None, :, None, :]

            def rotate(t):
                left, right = np.split(t, 2, axis=-1)
                return t * np.cos(angles) + np.concatenate([-right, left], -1) * np.sin(
                    angles
                )

            q, k = rotate(q), rotate(k)
        repeat = config.num_attention_heads // config.num_key_value_heads
        q = q.transpose(0, 2, 1, 3)
        k, v = (np.repeat(t, repeat, axis=2).transpose(0, 2, 1, 3) for t in (k, v))
        scores = (q @ k.swapaxes(-2, -1)) / np.sqrt(config.head_dim)
        allowed = positions[:, None] >= positions[None, :]
        if config.layer_types[i] == "sliding_attention" and config.sliding_window:
            allowed &= positions[:, None] - positions[None, :] < config.sliding_window
        scores = np.where(allowed, scores, -np.inf)
        sink = np.broadcast_to(
            weights[a + ".softmax_offset"][None, :, None, None], (*scores.shape[:-1], 1)
        )
        scores = np.concatenate([scores, sink], axis=-1)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        attention = (
            (probs[..., :-1] @ v).transpose(0, 2, 1, 3).reshape(batch, length, -1)
        )
        x = x + linear(attention, a + ".o_proj")
        hidden = norm(x, prefix + ".post_attention_layernorm")
        p = prefix + ".mlp"
        if i in config.mlp_only_layers:
            x = x + mlp(hidden, p)
            continue
        scores = 1 / (1 + np.exp(-linear(hidden, p + ".gate")))
        indices = np.argsort(scores + weights[p + ".expert_bias"], axis=-1)[
            ..., -config.num_experts_per_tok :
        ]
        route_weights = np.take_along_axis(scores, indices, axis=-1)
        route_weights /= route_weights.sum(axis=-1, keepdims=True) + 1e-20
        routed = np.zeros_like(hidden)
        for b in range(batch):
            for t in range(length):
                for e, score in zip(indices[b, t], route_weights[b, t]):
                    gate = (
                        hidden[b, t] @ weights[p + ".switch_mlp.gate_proj.weight"][e].T
                    )
                    up = hidden[b, t] @ weights[p + ".switch_mlp.up_proj.weight"][e].T
                    routed[b, t] += score * (
                        (gate / (1 + np.exp(-gate)) * up)
                        @ weights[p + ".switch_mlp.down_proj.weight"][e].T
                    )
        shared = (
            mlp(hidden, p + ".shared_expert")
            if config.shared_expert_intermediate_size
            else 0
        )
        x = x + routed + shared
    return linear(norm(x, "model.norm"), "lm_head")

def main():
    c = dict(model_type="flint", hidden_size=16, intermediate_size=24,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, vocab_size=32, rms_norm_eps=1e-6, max_position_embeddings=262144,
        rope_theta=10000.0, layer_types=["full_attention", "sliding_attention", "full_attention"],
        no_rope_layer_indices=[0,2], mlp_only_layers=[0], sliding_window=513,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=12,
        shared_expert_intermediate_size=20, tie_word_embeddings=False,
        hidden_act="silu", router_score_function="sigmoid", softmax_type="learnable",
        norm_topk_prob=True, attention_bias=False, use_sliding_window=True, decoder_sparse_step=1)
    rng=np.random.default_rng(20261021)
    w={}
    def weight(name,shape): w[name]=rng.normal(0,0.12,shape).astype(np.float32)
    def norm(name,n): w[name]=rng.uniform(.8,1.2,n).astype(np.float32)
    def mlp(p,inter,experts=None):
        batch=() if experts is None else (experts,)
        for name,shape in [("gate_proj",(inter,16)),("up_proj",(inter,16)),("down_proj",(16,inter))]:
            weight(p+"."+name+".weight",batch+shape)
    weight("model.embed_tokens.weight",(32,16)); weight("lm_head.weight",(32,16)); norm("model.norm.weight",16)
    for i in range(3):
        p=f"model.layers.{i}"
        for name,shape in [("q_proj",(32,16)),("k_proj",(16,16)),("v_proj",(16,16)),("o_proj",(16,32))]:
            weight(p+".self_attn."+name+".weight",shape)
        for n in ["q_norm","k_norm"]: norm(p+".self_attn."+n+".weight",8)
        w[p+".self_attn.softmax_offset"]=np.array([-2,0,1,3],np.float32)
        for n in ["input_layernorm","post_attention_layernorm"]: norm(p+"."+n+".weight",16)
        if i==0: mlp(p+".mlp",24)
        else:
            weight(p+".mlp.gate.weight",(4,16));w[p+".mlp.expert_bias"]=np.array([.05,-.04,.01,.03],np.float32)
            mlp(p+".mlp.switch_mlp",12,4); mlp(p+".mlp.shared_expert",20)
    ids=np.array([[1,4,7,2,9,3,8,5,6],[7,3,8,5,6,1,2,9,4]])
    logits=reference(SimpleNamespace(**c),{k:v.astype(np.float64) for k,v in w.items()},ids)
    out=dict(config=c,weights={k:dict(shape=list(v.shape),values=v.ravel().tolist()) for k,v in w.items()},
        ids=ids.tolist(),logits=logits.tolist(),oracle="NumPy float64, explicit per-expert loops; seed 20261021",
        atol=1e-3,rtol=1e-3)
    dest=Path(__file__).resolve().parents[1]/"Tests/MLXLMTests/Resources/flint-oracle.json"
    dest.write_text(json.dumps(out,separators=(",",":"))+"\n")
    print(dest)
if __name__ == "__main__": main()
