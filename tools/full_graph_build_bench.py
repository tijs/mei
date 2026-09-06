"""Apples-to-apples: how long does building the WHOLE qwen3_5_moe decode graph
take, structurally, versus Swift's measured decode.model_forward = 4.693 ms?

Graph-build cost scales with OP COUNT, not tensor size, so this uses tiny
tensors with the real op STRUCTURE (30 GDN + 10 attention + 40 MoE + shared
experts + lm_head). That keeps memory near zero and submits no GPU work, so it
is safe to run while another workload owns the GPU.

Purpose: separate "MLX graph construction is inherently expensive" from
"the Swift binding layer is expensive". Those have very different fixes.
"""
import time, mlx.core as mx

N_GDN, N_ATTN, N_MOE = 30, 10, 40
d = 64                      # tiny stand-in dim; op COUNT is what matters
ops = 0

def lin(x, w):
    global ops; ops += 1
    return x @ w

W = {k: mx.random.normal((d, d)).astype(mx.bfloat16) for k in
     ("qkv","z","a","b","out","q","k","v","o","g","u","dn","sg","su","sd","lm")}
Wb = mx.random.normal((8, d, d)).astype(mx.bfloat16)
x0 = mx.random.normal((1,1,d)).astype(mx.bfloat16)
mx.eval(*W.values(), Wb, x0)

def build():
    global ops; ops = 0
    x = x0
    for _ in range(N_GDN):                       # GDN layer
        ops += 1                                  # rms norm
        qkv = lin(x,W["qkv"]); z = lin(x,W["z"]); a = lin(x,W["a"]); b = lin(x,W["b"])
        g = mx.sigmoid(a) * mx.tanh(b); ops += 3  # gating
        c = qkv * g; ops += 1                     # conv/recurrence stand-in
        ops += 4                                  # scan: decay, state update, readout, norm
        x = lin(c + z, W["out"]); ops += 1
        ops += 1                                  # residual add
    for _ in range(N_ATTN):                      # attention layer
        ops += 1
        q = lin(x,W["q"]); k = lin(x,W["k"]); v = lin(x,W["v"])
        ops += 4                                  # rope q, rope k, q_norm, k_norm
        s = q * k; ops += 1                       # sdpa stand-in
        x = lin(s + v, W["o"]); ops += 2
    for _ in range(N_MOE):                       # MoE block
        ops += 1                                  # post-attn norm
        r = lin(x, W["g"]); ops += 3              # router: gate, softmax, argpartition
        ops += 2                                  # takealong, normalize
        e = mx.gather_mm(mx.expand_dims(mx.expand_dims(x,-2),-3), Wb,
                         rhs_indices=mx.array([[[0,1,2,3,4,5,6,7]]],dtype=mx.uint32)); ops += 1
        e2 = mx.gather_mm(e, Wb, rhs_indices=mx.array([[[0,1,2,3,4,5,6,7]]],dtype=mx.uint32)); ops += 1
        act = mx.sigmoid(e2)*e2; ops += 2
        e3 = mx.gather_mm(act, Wb, rhs_indices=mx.array([[[0,1,2,3,4,5,6,7]]],dtype=mx.uint32)); ops += 1
        comb = mx.sum(e3, axis=-3); ops += 2      # weight by scores, sum
        sg = lin(x,W["sg"]); su = lin(x,W["su"])  # shared expert
        sh = lin(mx.sigmoid(sg)*su, W["sd"]); ops += 2
        x = mx.squeeze(comb, -2) + sh + r*0; ops += 3
    ops += 1                                      # final norm
    x = lin(x, W["lm"])
    ops += 2                                      # sample
    return x

for _ in range(5): build()
t0 = time.perf_counter()
for _ in range(30): y = build()
ms = (time.perf_counter()-t0)/30*1000

print(f"mlx {mx.__version__}  (graph construction only, no eval, no GPU work)\n")
print(f"ops in one decode step (structural estimate): {ops}")
print(f"python-MLX graph build:  {ms:.3f} ms/token   ({ms*1000/ops:.2f} us/op)")
print()
print(f"Swift decode.model_forward (A2, measured):  4.693 ms/token")
print(f"ratio: Swift is {4.693/ms:.1f}x the python graph-build cost for an equivalent graph")
print()
print(f"If Swift matched python's per-op cost, model_forward would be ~{ms:.2f} ms,")
print(f"taking the decode step 16.220 -> {11.527+ms:.2f} ms = {1000/(11.527+ms):.1f} tok/s "
      f"(from 61.7, {(1000/(11.527+ms))/61.7:.2f}x)")
