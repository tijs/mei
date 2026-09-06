"""Offline model of a full qwen3_5_moe decode step, per component, vs tokens/forward.

Reproduces the decode step's LINEAR-PROJECTION work at the exact staged shapes
(Ornith 1.5 / Qwen 3.6 are identical here) so each component's cost and its
scaling with T (tokens verified per forward) can be measured without loading
the 20 GB model.

Modelled: all quantized projections in the 30 GDN layers, the 10 full-attention
layers, the 40 MoE blocks, and lm_head; plus the GDN depthwise conv.
NOT modelled: the GDN recurrent scan itself, SDPA over the KV cache, norms,
sampling. Those are elementwise/small-matmul work, so the total here is a
LOWER BOUND on the real step -- the gap to the measured 18.18 ms is the budget
those unmodelled parts consume.

T sweep answers the speculative-decoding question (Eagle3 / DFlash / DSpark /
native MTP): dense components should be near-flat in T, the MoE block is not.
"""
import time, mlx.core as mx

D, GS, BITS = 2048, 64, 4
N_GDN, N_ATTN, N_MOE = 30, 10, 40
E, H, K = 256, 512, 8
VOCAB = 248320

def qw(o, i):
    w = mx.random.normal((o, i)).astype(mx.bfloat16)
    a, s, b = mx.quantize(w, group_size=GS, bits=BITS); del w
    return a, s.astype(mx.bfloat16), b.astype(mx.bfloat16)

def qbank(o, i):
    w = mx.random.normal((E, o, i)).astype(mx.bfloat16)
    a, s, b = mx.quantize(w, group_size=GS, bits=BITS); del w
    return a, s.astype(mx.bfloat16), b.astype(mx.bfloat16)

def bench(fn, warmup=3, iters=12):
    for _ in range(warmup): mx.eval(fn())
    mx.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

qmm = mx.quantized_matmul
kw = dict(transpose=True, group_size=GS, bits=BITS)

print(f"mlx {mx.__version__}  device={mx.default_device()}")
print("qwen3_5_moe decode step, linear-projection model at staged shapes\n")

# --- GDN layer projections (shapes read from the staged checkpoint) ---
gdn = {n: qw(o, D) for n, o in (("in_proj_qkv", 8192), ("in_proj_z", 4096),
                                ("in_proj_a", 32), ("in_proj_b", 32))}
gdn["out_proj"] = qw(D, 4096)
conv = mx.random.normal((8192, 4, 1)).astype(mx.bfloat16)
# --- attention layer projections ---
att = {"q": qw(8192, D), "k": qw(512, D), "v": qw(512, D), "o": qw(D, 4096)}
# --- MoE routed bank ---
mg, ms_, mb = qbank(H, D); mu, us_, ub = qbank(H, D); md, ds_, db = qbank(D, H)
# --- shared expert + lm_head ---
sh = {"g": qw(H, D), "u": qw(H, D), "d": qw(D, H)}
lm = qw(VOCAB, D)
mx.eval(*[t for v in gdn.values() for t in v], conv,
        *[t for v in att.values() for t in v], mg, ms_, mb, mu, us_, ub, md, ds_, db,
        *[t for v in sh.values() for t in v], *lm)
print(f"resident: {mx.get_active_memory()/2**30:.2f} GiB\n")

def comps(T):
    x = mx.random.normal((1, T, D)).astype(mx.bfloat16)
    xm = mx.random.normal((1, T, 1, 1, D)).astype(mx.bfloat16)
    mx.random.seed(0)
    idx = mx.stack([mx.random.permutation(E)[:K] for _ in range(T)])[None].astype(mx.uint32)
    xc = mx.random.normal((1, T + 3, 8192)).astype(mx.bfloat16)
    mx.eval(x, xm, idx, xc)

    def gdn_layer():
        y = x
        for _ in range(N_GDN):
            qkv = qmm(y, *gdn["in_proj_qkv"], **kw)
            z = qmm(y, *gdn["in_proj_z"], **kw)
            a = qmm(y, *gdn["in_proj_a"], **kw); b = qmm(y, *gdn["in_proj_b"], **kw)
            c = mx.conv1d(xc, conv, groups=8192)
            y = qmm(z, *gdn["out_proj"], **kw)
            y = y + mx.sum(qkv[..., :D]) * 0 + mx.sum(a) * 0 + mx.sum(b) * 0 + mx.sum(c) * 0
        return y
    def attn_layer():
        y = x
        for _ in range(N_ATTN):
            q = qmm(y, *att["q"], **kw); k = qmm(y, *att["k"], **kw); v = qmm(y, *att["v"], **kw)
            y = qmm(q[..., :4096], *att["o"], **kw)
            y = y + mx.sum(k) * 0 + mx.sum(v) * 0
        return y
    def moe_block():
        y = xm
        for _ in range(N_MOE):
            g = mx.gather_qmm(y, mg, ms_, mb, rhs_indices=idx, **kw)
            u = mx.gather_qmm(y, mu, us_, ub, rhs_indices=idx, **kw)
            o = mx.gather_qmm(mx.sigmoid(g) * g * u, md, ds_, db, rhs_indices=idx, **kw)
            y = mx.sum(o, axis=-3, keepdims=True)
        return y
    def shared_expert():
        y = x
        for _ in range(N_MOE):
            g = qmm(y, *sh["g"], **kw); u = qmm(y, *sh["u"], **kw)
            y = qmm(mx.sigmoid(g) * g * u, *sh["d"], **kw)
        return y
    def lm_head():
        return qmm(x, *lm, **kw)
    return [("GDN proj x30", gdn_layer), ("full-attn proj x10", attn_layer),
            ("MoE routed x40", moe_block), ("shared expert x40", shared_expert),
            ("lm_head", lm_head)]

rows = {}
for T in (1, 2, 4, 8):
    rows[T] = [(n, bench(f)) for n, f in comps(T)]

names = [n for n, _ in rows[1]]
print(f"{'component':22s}" + "".join(f"{'T='+str(T):>12s}" for T in rows) + f"{'scaling 1->8':>14s}")
print("-" * 88)
for i, n in enumerate(names):
    vals = [rows[T][i][1] for T in rows]
    print(f"{n:22s}" + "".join(f"{v:12.3f}" for v in vals) + f"{vals[-1]/vals[0]:13.2f}x")
tot = [sum(v for _, v in rows[T]) for T in rows]
print("-" * 88)
print(f"{'TOTAL (modelled)':22s}" + "".join(f"{v:12.3f}" for v in tot) + f"{tot[-1]/tot[0]:13.2f}x")
print(f"{'per token':22s}" + "".join(f"{v/T:12.3f}" for T, v in zip(rows, tot)))
print(f"\nmeasured real decode step: 18.18 ms/token (55.0 tok/s)")
print(f"modelled T=1 total: {tot[0]:.2f} ms  ->  unmodelled remainder: {18.18-tot[0]:.2f} ms")
print(f"\nper-token speedup from verifying T at once (modelled parts only):")
for T, v in zip(rows, tot):
    print(f"   T={T}: {tot[0]/(v/T):.2f}x")
