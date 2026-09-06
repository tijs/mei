"""Offline proxy for P1's ceiling: what does mx.compile actually buy on the
qwen3_5_moe decode step?

vmlx's VMLX_ENABLE_UNSAFE_COMPILE gate turns on shapeless compile across the
decode path; the upstream comment claims +45-70%. This measures the same class
of change on the modelled decode step (all quantized projections at the staged
shapes, 40 MoE blocks + 30 GDN + 10 attn + shared experts + lm_head), comparing
eager vs one compiled region per component.

This is a PROXY, not the real thing -- it compiles the projection graph, while
vmlx compiles the true decode step including norms, rope, the GDN scan and
sampling. Treat it as an order-of-magnitude expectation for P1, not a forecast.
"""
import time, mlx.core as mx

D, GS, BITS, E, H, K, VOCAB = 2048, 64, 4, 256, 512, 8, 248320
N_GDN, N_ATTN, N_MOE = 30, 10, 40

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
    mx.synchronize(); return (time.perf_counter()-t0)/iters*1000

qmm, kw = mx.quantized_matmul, dict(transpose=True, group_size=GS, bits=BITS)
gdn = {n: qw(o, D) for n, o in (("qkv",8192),("z",4096),("a",32),("b",32))}
gdn["out"] = qw(D, 4096)
att = {"q": qw(8192,D), "k": qw(512,D), "v": qw(512,D), "o": qw(D,4096)}
mg,ms_,mb = qbank(H,D); mu,us_,ub = qbank(H,D); md,ds_,db = qbank(D,H)
sh = {"g": qw(H,D), "u": qw(H,D), "d": qw(D,H)}
lm = qw(VOCAB, D)
mx.eval(*[t for v in gdn.values() for t in v], *[t for v in att.values() for t in v],
        mg,ms_,mb,mu,us_,ub,md,ds_,db, *[t for v in sh.values() for t in v], *lm)

x = mx.random.normal((1,1,D)).astype(mx.bfloat16)
xm = mx.random.normal((1,1,1,1,D)).astype(mx.bfloat16)
mx.random.seed(0)
idx = mx.random.permutation(E)[:K][None,None].astype(mx.uint32)
mx.eval(x, xm, idx)

def gdn_body(y):
    qkv = qmm(y, *gdn["qkv"], **kw); z = qmm(y, *gdn["z"], **kw)
    a = qmm(y, *gdn["a"], **kw); b = qmm(y, *gdn["b"], **kw)
    g = mx.sigmoid(a) * mx.tanh(b)                       # gating, elementwise
    o = qmm(z * mx.sigmoid(z), *gdn["out"], **kw)        # gated out-projection
    return o + mx.sum(qkv)*0 + mx.sum(g)*0
def attn_body(y):
    q = qmm(y, *att["q"], **kw); k = qmm(y, *att["k"], **kw); v = qmm(y, *att["v"], **kw)
    return qmm(q[..., :4096] * mx.sigmoid(q[..., :4096]), *att["o"], **kw) + mx.sum(k)*0 + mx.sum(v)*0
def moe_body(y):
    g = mx.gather_qmm(y, mg, ms_, mb, rhs_indices=idx, **kw)
    u = mx.gather_qmm(y, mu, us_, ub, rhs_indices=idx, **kw)
    o = mx.gather_qmm(mx.sigmoid(g)*g*u, md, ds_, db, rhs_indices=idx, **kw)
    return mx.sum(o, axis=-3, keepdims=True)
def shared_body(y):
    g = qmm(y, *sh["g"], **kw); u = qmm(y, *sh["u"], **kw)
    return qmm(mx.sigmoid(g)*g*u, *sh["d"], **kw)

cg, ca, cm, cs = map(mx.compile, (gdn_body, attn_body, moe_body, shared_body))
def chain(body, n, seed): 
    def r():
        y = seed
        for _ in range(n): y = body(y)
        return y
    return r

legs = [("GDN proj x30", chain(gdn_body,N_GDN,x), chain(cg,N_GDN,x)),
        ("full-attn proj x10", chain(attn_body,N_ATTN,x), chain(ca,N_ATTN,x)),
        ("MoE routed x40", chain(moe_body,N_MOE,xm), chain(cm,N_MOE,xm)),
        ("shared expert x40", chain(shared_body,N_MOE,x), chain(cs,N_MOE,x)),
        ("lm_head", lambda: qmm(x,*lm,**kw), mx.compile(lambda: qmm(x,*lm,**kw)))]

print(f"mlx {mx.__version__}  device={mx.default_device()}\n")
print(f"{'component':22s} {'eager ms':>10s} {'compiled ms':>12s} {'speedup':>9s}")
print("-"*58)
te = tc = 0.0
for n, fe, fc in legs:
    e, c = bench(fe), bench(fc); te += e; tc += c
    print(f"{n:22s} {e:10.3f} {c:12.3f} {e/c:8.2f}x")
print("-"*58)
print(f"{'TOTAL (modelled)':22s} {te:10.3f} {tc:12.3f} {te/tc:8.2f}x")
rem = 18.18 - te
print(f"\nunmodelled remainder held constant at {rem:.2f} ms (GDN scan, SDPA, norms, sampling)")
print(f"implied full step: {te+rem:.2f} -> {tc+rem:.2f} ms  =  {(te+rem)/(tc+rem):.2f}x  "
      f"({1000/(te+rem):.1f} -> {1000/(tc+rem):.1f} tok/s)")
print(f"if the remainder compiled equally well: {te/tc:.2f}x -> {1000/((te+rem)/(te/tc)):.1f} tok/s")
