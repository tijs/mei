"""Decode-shaped MoE microbenchmark for qwen3_5_moe (Ornith 1.5 / Qwen 3.6).

Tests the F4/F5 levers WITHOUT loading the 20 GB model: builds ONE layer's
routed-expert bank at the real shapes and times the decode-shaped
gather_qmm patterns that SwitchGLU actually issues.

Weights are random; only the SHAPES and dtypes match the staged checkpoint.
That is sufficient: we are measuring dispatch/kernel cost, not numerics.
"""
import time, argparse
import mlx.core as mx

E, D, H, K = 256, 2048, 512, 8      # experts, hidden, moe_intermediate, top-k
GS, BITS = 64, 4
LAYERS = 40

def qbank(out_dim, in_dim):
    w = mx.random.normal((E, out_dim, in_dim)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=GS, bits=BITS)
    del w
    return wq, s.astype(mx.bfloat16), b.astype(mx.bfloat16)

def bench(fn, warmup=5, iters=30):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0   # ms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()

    print(f"mlx {mx.__version__}  device={mx.default_device()}")
    print(f"shapes: E={E} D={D} H={H} top_k={K} bits={BITS} group={GS}\n")

    gw, gs, gb = qbank(H, D)
    uw, us, ub = qbank(H, D)
    dw, ds, db = qbank(D, H)
    # pre-fused gate+up bank (the F5 repack layout: [E, 2H, in_packed])
    fw = mx.concatenate([gw, uw], axis=-2)
    fs = mx.concatenate([gs, us], axis=-2)
    fb = mx.concatenate([gb, ub], axis=-2)
    mx.eval(gw, gs, gb, uw, us, ub, dw, ds, db, fw, fs, fb)
    resident = mx.get_active_memory() / 2**20
    print(f"resident after build (incl. fused copy): {resident:.0f} MiB\n")

    x = mx.random.normal((1, 1, 1, 1, D)).astype(mx.bfloat16)
    idx = mx.array([[list(range(K))]], dtype=mx.uint32)   # [1,1,K]
    mx.eval(x, idx)

    qmm = mx.gather_qmm
    kw = dict(transpose=True, group_size=GS, bits=BITS)

    def split_path():                      # CURRENT production path (fuse cache off)
        g = qmm(x, gw, gs, gb, rhs_indices=idx, **kw)
        u = qmm(x, uw, us, ub, rhs_indices=idx, **kw)
        act = mx.sigmoid(g) * g * u        # silu(g)*u
        return qmm(act, dw, ds, db, rhs_indices=idx, **kw)

    def fused_gateup_path():               # F5: pre-fused gate+up bank
        c = qmm(x, fw, fs, fb, rhs_indices=idx, **kw)
        g, u = mx.split(c, 2, axis=-1)
        act = mx.sigmoid(g) * g * u
        return qmm(act, dw, ds, db, rhs_indices=idx, **kw)

    compiled_split = mx.compile(split_path)      # F4: trusted compiled region
    compiled_fused = mx.compile(fused_gateup_path)

    legs = [
        ("split 3x gather_qmm (CURRENT prod path)", split_path),
        ("pre-fused gate+up, 2x gather_qmm (F5)", fused_gateup_path),
        ("split, mx.compile'd region (F4)", compiled_split),
        ("pre-fused + compiled (F4+F5)", compiled_fused),
    ]
    base = None
    print(f"{'leg':46s} {'ms/layer':>9s} {'x40 layers':>11s} {'vs base':>8s}")
    print("-" * 78)
    for name, fn in legs:
        ms = bench(fn, iters=a.iters)
        if base is None:
            base = ms
        print(f"{name:46s} {ms:9.4f} {ms*LAYERS:9.2f}ms {base/ms:7.2f}x")

    print(f"\nMeasured full-model decode: 18.18 ms/token (55.0 tok/s)")
    print(f"MoE-only x40 above is the routed-expert share of that budget.")

if __name__ == "__main__":
    main()
