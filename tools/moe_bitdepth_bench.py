"""Does routed-expert bit depth cost decode SPEED on qwen3_5_moe?

If decode is dispatch/kernel-bound rather than bandwidth-bound (see F2), then
lowering bits should buy memory but NOT much speed -- inverting the usual
"lower bits = faster" heuristic that drove this project's GGUF quant choices.
Chained 40-layer decode-shaped MoE, one eval per token, same as
moe_chain_bench.py.
"""
import time
import mlx.core as mx

E, D, H, K, GS, LAYERS = 256, 2048, 512, 8, 64, 40

def qbank(out_dim, in_dim, bits):
    w = mx.random.normal((E, out_dim, in_dim)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=GS, bits=bits)
    del w
    return wq, s.astype(mx.bfloat16), b.astype(mx.bfloat16)

def bench(fn, warmup=3, iters=15):
    for _ in range(warmup): mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0

print(f"mlx {mx.__version__}  device={mx.default_device()}")
print(f"qwen3_5_moe routed block: E={E} D={D} H={H} top_k={K} group={GS}, {LAYERS} layers chained\n")
print(f"{'bits':>5s} {'bank GiB(40L)':>14s} {'active MiB/tok':>15s} {'ms/token':>9s} {'vs 4-bit':>9s}")
print("-" * 62)

x0 = mx.random.normal((1, 1, 1, 1, D)).astype(mx.bfloat16)
idx = mx.array([[list(range(K))]], dtype=mx.uint32)
mx.eval(x0, idx)
base = None
for bits in (2, 3, 4, 5, 6, 8):
    gw, gs, gb = qbank(H, D, bits); uw, us, ub = qbank(H, D, bits); dw, ds, db = qbank(D, H, bits)
    mx.eval(gw, gs, gb, uw, us, ub, dw, ds, db)
    kw = dict(transpose=True, group_size=GS, bits=bits)
    qmm = mx.gather_qmm
    def run():
        x = x0
        for _ in range(LAYERS):
            g = qmm(x, gw, gs, gb, rhs_indices=idx, **kw)
            u = qmm(x, uw, us, ub, rhs_indices=idx, **kw)
            y = qmm(mx.sigmoid(g) * g * u, dw, ds, db, rhs_indices=idx, **kw)
            x = mx.sum(y, axis=-3, keepdims=True)
        return x
    ms = bench(run)
    if base is None or bits == 4: base = base if bits != 4 else ms
    # per-layer bank bytes: (gate+up: 2*H*D + down: D*H) * bits/8 * E  + scales/biases
    wbytes = E * (2*H*D + D*H) * bits / 8
    sbbytes = E * (2*H*D + D*H) / GS * 2 * 2       # scales+biases, bf16
    bank40 = (wbytes + sbbytes) * LAYERS / 2**30
    active = (wbytes + sbbytes) * K / E * LAYERS / 2**20
    print(f"{bits:5d} {bank40:13.2f}G {active:14.0f}M {ms:9.3f} {'' if base is None else f'{base/ms:8.2f}x'}")
    del gw, gs, gb, uw, us, ub, dw, ds, db
    mx.clear_cache()
print("\nnote: 4-bit row is the current production quant (19.5 GB staged checkpoint)")
