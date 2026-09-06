"""How much CPU time does BUILDING the decode graph cost, per op eliminated?

A2 showed decode.model_forward = 4.693 ms/token of CPU-side lazy graph
construction, with the GPU idle for all of it -- 28.9% of the step. That makes
op-count reduction doubly valuable: every op removed saves both its GPU dispatch
AND its graph-build cost.

This measures the graph-build half ONLY: build the lazy graph and never eval it.
No GPU work is submitted, so this is safe to run while another workload owns the
GPU. Python-MLX graph-build overhead is not Swift's, so read the PER-OP cost and
the RATIOS, not the absolute ms.
"""
import time, mlx.core as mx

E, D, H, K, GS, BITS, LAYERS = 256, 2048, 512, 8, 64, 4, 40

def qbank(o, i):
    w = mx.random.normal((E, o, i)).astype(mx.bfloat16)
    a, s, b = mx.quantize(w, group_size=GS, bits=BITS); del w
    return a, s.astype(mx.bfloat16), b.astype(mx.bfloat16)

gw,gs_,gb = qbank(H,D); uw,us_,ub = qbank(H,D); dw,ds_,db = qbank(D,H)
fw = mx.concatenate([gw,uw], axis=-2); fs = mx.concatenate([gs_,us_], axis=-2); fb = mx.concatenate([gb,ub], axis=-2)
x0 = mx.random.normal((1,1,1,1,D)).astype(mx.bfloat16)
idx = mx.random.permutation(E)[:K][None,None].astype(mx.uint32)
mx.eval(gw,gs_,gb,uw,us_,ub,dw,ds_,db,fw,fs,fb,x0,idx)   # one-off, before timing
kw = dict(transpose=True, group_size=GS, bits=BITS)
qmm = mx.gather_qmm

def build_split(n):
    x = x0
    for _ in range(n):
        g = qmm(x, gw,gs_,gb, rhs_indices=idx, **kw)
        u = qmm(x, uw,us_,ub, rhs_indices=idx, **kw)
        y = qmm(mx.sigmoid(g)*g*u, dw,ds_,db, rhs_indices=idx, **kw)
        x = mx.sum(y, axis=-3, keepdims=True)
    return x

def build_fused(n):
    x = x0
    for _ in range(n):
        c = qmm(x, fw,fs,fb, rhs_indices=idx, **kw)
        g, u = mx.split(c, 2, axis=-1)
        y = qmm(mx.sigmoid(g)*g*u, dw,ds_,db, rhs_indices=idx, **kw)
        x = mx.sum(y, axis=-3, keepdims=True)
    return x

# op count per layer, counted from the source above
OPS = {"split (3 qmm + sigmoid + 2 mul + sum)": 7, "fused (2 qmm + split + sigmoid + 2 mul + sum)": 7}

def time_build(fn, n, iters=40):
    for _ in range(5): fn(n)          # warm
    t0 = time.perf_counter()
    for _ in range(iters): fn(n)
    return (time.perf_counter()-t0)/iters*1000

print(f"mlx {mx.__version__}  (graph build only -- no eval, no GPU submission)\n")
print(f"{'layers':>7s} {'split ms':>10s} {'fused ms':>10s} {'delta':>8s}")
print("-"*40)
prev=None
for n in (10, 20, 40, 80):
    a = time_build(build_split, n); b = time_build(build_fused, n)
    print(f"{n:7d} {a:10.3f} {b:10.3f} {a-b:8.3f}")
    if n == 40: at40 = (a, b)

# marginal cost per layer, from the 40->80 slope
a80 = time_build(build_split, 80); a40 = time_build(build_split, 40)
per_layer = (a80-a40)/40
print(f"\nmarginal graph-build cost: {per_layer*1000:.1f} us per MoE layer "
      f"({per_layer*1000/7:.1f} us per op, 7 ops/layer)")
print(f"40-layer MoE graph build: split {at40[0]:.3f} ms, fused {at40[1]:.3f} ms "
      f"-> fusing gate+up saves {at40[0]-at40[1]:.3f} ms of CPU per token")
print(f"\nA2 measured decode.model_forward = 4.693 ms/token (Swift, whole model).")
print("Python overhead differs from Swift; use the per-op figure and the ratio.")
