"""Does the ORDER of routed expert indices change decode cost?

A contended first run suggested sorted indices were ~1.35x faster than an
unsorted permutation at the same top-8 shape. That test had a confound: it
compared one specific permutation against its own sorted version, so "sorted"
and "which experts" were not separated, and the machine was busy.

This isolates it: N independent random top-8 draws, each timed BOTH unsorted and
sorted (identical expert sets, only the order differs), plus a contiguous 0..7
best-case locality reference and a maximally-spread reference. Reports the
paired mean, so expert-set effects cancel.

Refuses to run if another inference workload is live.
"""
import time, sys, subprocess, statistics
import mlx.core as mx

E, D, H, K, GS, BITS, LAYERS = 256, 2048, 512, 8, 64, 4, 40

def busy():
    out = subprocess.run(["ps", "ax", "-o", "command"], capture_output=True, text=True).stdout
    return [l for l in out.splitlines()
            if any(k in l for k in ("llama-server", "/release/mei", "probe_mei", "mlx_lm"))
            and "grep" not in l]

def qbank(o, i):
    w = mx.random.normal((E, o, i)).astype(mx.bfloat16)
    a, s, b = mx.quantize(w, group_size=GS, bits=BITS); del w
    return a, s.astype(mx.bfloat16), b.astype(mx.bfloat16)

def bench(fn, warmup=3, iters=12):
    for _ in range(warmup): mx.eval(fn())
    mx.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): mx.eval(fn())
    mx.synchronize(); return (time.perf_counter()-t0)/iters*1000

b = busy()
if b:
    print("REFUSING: inference workload live:"); [print("  ", l[:100]) for l in b[:3]]
    sys.exit(2)

print(f"mlx {mx.__version__}  device={mx.default_device()}")
print(f"E={E} D={D} H={H} top_k={K} bits={BITS} group={GS}, {LAYERS} layers chained\n")
gw,gs_,gb = qbank(H,D); uw,us_,ub = qbank(H,D); dw,ds_,db = qbank(D,H)
x0 = mx.random.normal((1,1,1,1,D)).astype(mx.bfloat16)
mx.eval(gw,gs_,gb,uw,us_,ub,dw,ds_,db,x0)
kw = dict(transpose=True, group_size=GS, bits=BITS)

def run_with(idx):
    def r():
        x = x0
        for _ in range(LAYERS):
            g = mx.gather_qmm(x, gw,gs_,gb, rhs_indices=idx, **kw)
            u = mx.gather_qmm(x, uw,us_,ub, rhs_indices=idx, **kw)
            y = mx.gather_qmm(mx.sigmoid(g)*g*u, dw,ds_,db, rhs_indices=idx, **kw)
            x = mx.sum(y, axis=-3, keepdims=True)
        return x
    return r

N = 6
un, so = [], []
for t in range(N):
    mx.random.seed(100 + t)
    raw = mx.random.permutation(E)[:K]
    iu = raw[None,None].astype(mx.uint32)
    ist = mx.sort(raw)[None,None].astype(mx.uint32)
    mx.eval(iu, ist)
    a = bench(run_with(iu)); c = bench(run_with(ist))
    un.append(a); so.append(c)
    print(f"  draw {t+1}: unsorted {a:7.3f}   sorted {c:7.3f}   {a/c:5.2f}x")

print(f"\npaired mean: unsorted {statistics.mean(un):.3f} ms, sorted {statistics.mean(so):.3f} ms"
      f"  ->  {statistics.mean(un)/statistics.mean(so):.3f}x")
print(f"  (sd unsorted {statistics.stdev(un):.3f}, sorted {statistics.stdev(so):.3f})")

print("\nlocality references (same top-8 count, different spread):")
for name, arr in (("contiguous 0..7", list(range(K))),
                  ("stride-32 spread", [i*32 for i in range(K)]),
                  ("all-same expert 0", [0]*K)):
    idx = mx.array(arr, dtype=mx.uint32)[None,None]; mx.eval(idx)
    print(f"  {name:20s} {bench(run_with(idx)):7.3f} ms")
after = busy()
print(f"\nworkload check after run: {'CONTENDED - discard' if after else 'clean'}")
