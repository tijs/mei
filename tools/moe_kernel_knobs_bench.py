"""Two unmeasured kernel-level knobs on the qwen3_5_moe routed block.

(1) sorted_indices: production passes sortedIndices=false at decode because
    SwitchLayers' `doSort = indices.size >= 64` is false at top-8. Whether the
    sorted gather_qmm path is faster at DECODE shapes has never been measured.
(2) group_size: the checkpoint uses g64. g32/g128 change scale/bias volume and
    may hit different kernels, the same way bit depth did (2/4/8 fast, 3/5/6 slow).

40 layers chained, one eval, same harness as moe_chain_bench.py.
"""
import time, mlx.core as mx
import subprocess as _sp, sys as _sys
def _busy():
    o=_sp.run(["ps","ax","-o","command"],capture_output=True,text=True).stdout
    return [l for l in o.splitlines() if any(k in l for k in ("llama-server","/release/mei","probe_mei","mlx_lm")) and "grep" not in l]
_b=_busy()
if _b:
    print("REFUSING: inference workload live:"); [print("  ",l[:100]) for l in _b[:3]]; _sys.exit(2)



E, D, H, K, BITS, LAYERS = 256, 2048, 512, 8, 4, 40

def qbank(o, i, gs):
    w = mx.random.normal((E, o, i)).astype(mx.bfloat16)
    a, s, b = mx.quantize(w, group_size=gs, bits=BITS); del w
    return a, s.astype(mx.bfloat16), b.astype(mx.bfloat16)

def bench(fn, warmup=3, iters=15):
    for _ in range(warmup): mx.eval(fn())
    mx.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): mx.eval(fn())
    mx.synchronize(); return (time.perf_counter()-t0)/iters*1000

print(f"mlx {mx.__version__}  device={mx.default_device()}")
print(f"E={E} D={D} H={H} top_k={K} bits={BITS}, {LAYERS} layers chained\n")

x0 = mx.random.normal((1,1,1,1,D)).astype(mx.bfloat16)
mx.random.seed(0)
raw = mx.random.permutation(E)[:K]
idx_unsorted = raw[None,None].astype(mx.uint32)
idx_sorted   = mx.sort(raw)[None,None].astype(mx.uint32)
mx.eval(x0, idx_unsorted, idx_sorted)

print("--- (1) sorted_indices at decode shapes (group_size=64) ---")
gw,gs_,gb = qbank(H,D,64); uw,us_,ub = qbank(H,D,64); dw,ds_,db = qbank(D,H,64)
mx.eval(gw,gs_,gb,uw,us_,ub,dw,ds_,db)
def make(idx, sortedflag):
    kw = dict(transpose=True, group_size=64, bits=BITS, sorted_indices=sortedflag)
    def run():
        x = x0
        for _ in range(LAYERS):
            g = mx.gather_qmm(x, gw,gs_,gb, rhs_indices=idx, **kw)
            u = mx.gather_qmm(x, uw,us_,ub, rhs_indices=idx, **kw)
            y = mx.gather_qmm(mx.sigmoid(g)*g*u, dw,ds_,db, rhs_indices=idx, **kw)
            x = mx.sum(y, axis=-3, keepdims=True)
        return x
    return run
legs = [("unsorted idx, sorted_indices=False (PRODUCTION)", make(idx_unsorted, False)),
        ("sorted idx,   sorted_indices=True", make(idx_sorted, True)),
        ("sorted idx,   sorted_indices=False", make(idx_sorted, False))]
base=None
print(f"{'leg':48s} {'ms':>8s} {'vs prod':>9s}")
for n,f in legs:
    ms=bench(f)
    if base is None: base=ms
    print(f"  {n:46s} {ms:8.3f} {base/ms:8.2f}x")
del gw,gs_,gb,uw,us_,ub,dw,ds_,db; mx.clear_cache()

print("\n--- (2) group_size sweep (bits=4) ---")
print(f"{'group_size':>10s} {'scale+bias GiB(40L)':>21s} {'ms':>8s} {'vs g64':>8s}")
base=None
for gs in (32, 64, 128):
    gw,gsc,gb = qbank(H,D,gs); uw,usc,ub = qbank(H,D,gs); dw,dsc,db = qbank(D,H,gs)
    mx.eval(gw,gsc,gb,uw,usc,ub,dw,dsc,db)
    kw = dict(transpose=True, group_size=gs, bits=BITS)
    def run():
        x = x0
        for _ in range(LAYERS):
            g = mx.gather_qmm(x, gw,gsc,gb, rhs_indices=idx_unsorted, **kw)
            u = mx.gather_qmm(x, uw,usc,ub, rhs_indices=idx_unsorted, **kw)
            y = mx.gather_qmm(mx.sigmoid(g)*g*u, dw,dsc,db, rhs_indices=idx_unsorted, **kw)
            x = mx.sum(y, axis=-3, keepdims=True)
        return x
    ms = bench(run)
    if gs == 64: base = ms
    sb = E*(2*H*D + D*H)/gs*2*2*LAYERS/2**30
    print(f"{gs:10d} {sb:20.2f}G {ms:8.3f}" + (f" {base/ms:7.2f}x" if base else ""))
    del gw,gsc,gb,uw,usc,ub,dw,dsc,db; mx.clear_cache()
print("\nworkload check after run:", "CONTENDED - discard" if _busy() else "clean")
