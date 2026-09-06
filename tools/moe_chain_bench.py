"""Decode-shaped MoE microbenchmark, CHAINED — one graph, one eval.

Fixes the methodology error of moe_dispatch_bench.py, which called mx.eval()
per layer and so paid a full CPU<->GPU round-trip 40x that the real model
never pays (the real decode step builds all 40 layers lazily and evals once).

Weights: ONE layer's routed-expert bank at the real qwen3_5_moe shapes, reused
for all 40 chained calls. The bank is 432 MiB and M1 Max L2 is 48 MB, so reads
still miss to DRAM; but true per-layer weight traffic is understated, so treat
the result as a LOWER BOUND on the MoE share of the decode step.
"""
import time, argparse
import mlx.core as mx

E, D, H, K = 256, 2048, 512, 8
GS, BITS = 64, 4
LAYERS = 40

def qbank(out_dim, in_dim):
    w = mx.random.normal((E, out_dim, in_dim)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=GS, bits=BITS)
    del w
    return wq, s.astype(mx.bfloat16), b.astype(mx.bfloat16)

def bench(fn, warmup=3, iters=20):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--layers", type=int, default=LAYERS)
    a = ap.parse_args()
    L = a.layers

    print(f"mlx {mx.__version__}  device={mx.default_device()}  layers={L}")
    print(f"shapes: E={E} D={D} H={H} top_k={K} bits={BITS} group={GS}\n")

    gw, gs, gb = qbank(H, D); uw, us, ub = qbank(H, D); dw, ds, db = qbank(D, H)
    fw = mx.concatenate([gw, uw], axis=-2)
    fs = mx.concatenate([gs, us], axis=-2)
    fb = mx.concatenate([gb, ub], axis=-2)
    mx.eval(gw, gs, gb, uw, us, ub, dw, ds, db, fw, fs, fb)
    print(f"resident after build (incl. fused copy): {mx.get_active_memory()/2**20:.0f} MiB\n")

    x0 = mx.random.normal((1, 1, 1, 1, D)).astype(mx.bfloat16)
    idx = mx.array([[list(range(K))]], dtype=mx.uint32)
    mx.eval(x0, idx)
    qmm = mx.gather_qmm
    kw = dict(transpose=True, group_size=GS, bits=BITS)

    def split_layer(x):
        g = qmm(x, gw, gs, gb, rhs_indices=idx, **kw)
        u = qmm(x, uw, us, ub, rhs_indices=idx, **kw)
        act = mx.sigmoid(g) * g * u
        y = qmm(act, dw, ds, db, rhs_indices=idx, **kw)
        return mx.sum(y, axis=-3, keepdims=True)          # combine routed outputs

    def fused_layer(x):
        c = qmm(x, fw, fs, fb, rhs_indices=idx, **kw)
        g, u = mx.split(c, 2, axis=-1)
        act = mx.sigmoid(g) * g * u
        y = qmm(act, dw, ds, db, rhs_indices=idx, **kw)
        return mx.sum(y, axis=-3, keepdims=True)

    csplit = mx.compile(split_layer)
    cfused = mx.compile(fused_layer)

    def chain(layer_fn):
        def run():
            x = x0
            for _ in range(L):
                x = layer_fn(x)
            return x
        return run

    legs = [
        ("split 3x gather_qmm (CURRENT prod path)", chain(split_layer)),
        ("pre-fused gate+up (F5 repack)",           chain(fused_layer)),
        ("split, per-layer mx.compile (F4)",        chain(csplit)),
        ("pre-fused + compiled (F4+F5)",            chain(cfused)),
    ]
    base = None
    print(f"{'leg':44s} {'ms/token':>9s} {'vs base':>8s} {'implied tok/s*':>14s}")
    print("-" * 80)
    for name, fn in legs:
        ms = bench(fn, iters=a.iters)
        if base is None: base = ms
        print(f"{name:44s} {ms:9.3f} {base/ms:7.2f}x {1000/ms:13.1f}")
    print("\n* implied tok/s if the MoE blocks were the ONLY cost (they are not)")
    print("  measured full-model decode: 18.18 ms/token (55.0 tok/s)")

if __name__ == "__main__":
    main()
