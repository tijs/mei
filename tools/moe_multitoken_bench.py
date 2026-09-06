"""Speculative-decoding economics for qwen3_5_moe, measured offline.

A draft/verify scheme (Eagle3, DFlash, DSpark, native MTP) only pays off if
verifying T tokens in one forward costs much less than T sequential forwards.
On a DENSE model at the memory-bandwidth wall it roughly does -- weights are
streamed once regardless of T -- which is why llama.cpp saw the draft cost
exceed the win.

MoE changes the arithmetic in BOTH directions:
  - against: T tokens route to up to T*top_k DISTINCT experts, so routed weight
    traffic grows with T (unlike a dense model, where it is flat).
  - for: this family measured 3.8x BELOW its bandwidth ceiling, so there is
    slack to absorb that growth.

Which effect wins is an empirical question. This measures it at the real
shapes, with realistic per-token routing (distinct random experts per token),
40 layers chained into one graph.
"""
import time
import mlx.core as mx

E, D, H, K, GS, BITS, LAYERS = 256, 2048, 512, 8, 64, 4, 40

def qbank(o, i):
    w = mx.random.normal((E, o, i)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=GS, bits=BITS); del w
    return wq, s.astype(mx.bfloat16), b.astype(mx.bfloat16)

def bench(fn, warmup=3, iters=15):
    for _ in range(warmup): mx.eval(fn())
    mx.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

print(f"mlx {mx.__version__}  device={mx.default_device()}")
print(f"qwen3_5_moe routed block, {LAYERS} layers chained, top_k={K} of {E} experts\n")
gw, gs, gb = qbank(H, D); uw, us, ub = qbank(H, D); dw, ds, db = qbank(D, H)
mx.eval(gw, gs, gb, uw, us, ub, dw, ds, db)
kw = dict(transpose=True, group_size=GS, bits=BITS); qmm = mx.gather_qmm

print(f"{'tokens/fwd':>10s} {'ms/forward':>11s} {'ms/token':>9s} {'speedup vs T=1':>15s} {'distinct experts':>17s}")
print("-" * 70)
base = None
for T in (1, 2, 3, 4, 6, 8):
    mx.random.seed(0)
    # realistic routing: each token independently picks top_k of E
    idx = mx.stack([mx.random.permutation(E)[:K] for _ in range(T)])[None].astype(mx.uint32)
    x0 = mx.random.normal((1, T, 1, 1, D)).astype(mx.bfloat16)
    mx.eval(idx, x0)
    distinct = len(set(idx.reshape(-1).tolist()))
    def run():
        x = x0
        for _ in range(LAYERS):
            g = qmm(x, gw, gs, gb, rhs_indices=idx, **kw)
            u = qmm(x, uw, us, ub, rhs_indices=idx, **kw)
            y = qmm(mx.sigmoid(g) * g * u, dw, ds, db, rhs_indices=idx, **kw)
            x = mx.sum(y, axis=-3, keepdims=True)
        return x
    ms = bench(run)
    if base is None: base = ms
    print(f"{T:10d} {ms:11.3f} {ms/T:9.3f} {base/(ms/T):14.2f}x {distinct:17d}")
print("\nspeedup vs T=1 = how many times cheaper a token is when verified in a batch of T.")
print("A draft scheme wins if this exceeds (1 + draft_cost) / acceptance_rate.")
