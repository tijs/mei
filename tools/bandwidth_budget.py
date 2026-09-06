"""Per-decode-token weight-traffic budget for an MLX MoE checkpoint.

Reads safetensors headers only -- no weights loaded, no GPU, runs in ~1s.
Answers: is decode on this model bandwidth-bound, or overhead-bound?

usage: bandwidth_budget.py <staged-model-dir> [--topk 8] [--experts 256]
"""
import json, glob, struct, re, sys, collections, argparse, os

DS = {'BF16': 2, 'F16': 2, 'F32': 4, 'U32': 4, 'U8': 1, 'I32': 4, 'I16': 2}

def load_header(d):
    hdr = {}
    for f in sorted(glob.glob(os.path.join(d, '*.safetensors'))):
        with open(f, 'rb') as fh:
            n = struct.unpack('<Q', fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != '__metadata__':
                hdr[k] = v
    return hdr

def nbytes(v):
    n = 1
    for d in v['shape']:
        n *= d
    return n * DS[v['dtype']]

def classify(rest):
    if rest.startswith('mlp.switch_mlp'):        return 'moe_routed_bank'
    if rest.startswith('mlp.shared_expert_gate'): return 'shared_gate'
    if rest.startswith('mlp.shared_expert'):      return 'shared_expert'
    if rest.startswith('mlp.gate'):               return 'router_gate'
    if rest.startswith('linear_attn'):            return 'gdn_linear_attn'
    if rest.startswith('self_attn'):              return 'full_attn'
    return 'norms'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_dir')
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--experts', type=int, default=256)
    ap.add_argument('--measured-ms', type=float, default=None,
                    help='measured ms/token, to print the headroom ratio')
    a = ap.parse_args()

    hdr = load_header(a.model_dir)
    if not hdr:
        sys.exit(f'no safetensors in {a.model_dir}')
    total = sum(nbytes(v) for v in hdr.values())
    print(f'{a.model_dir}\n  total tensor bytes: {total/2**30:.2f} GiB  tensors: {len(hdr)}')

    per_layer = collections.defaultdict(lambda: collections.defaultdict(int))
    nonlayer = collections.defaultdict(int)
    for k, v in hdr.items():
        m = re.search(r'layers\.(\d+)\.(.*)', k)
        if m:
            per_layer[int(m.group(1))][classify(m.group(2))] += nbytes(v)
        else:
            nonlayer[k] += nbytes(v)

    lm = sum(b for k, b in nonlayer.items() if 'lm_head' in k)
    comp = collections.defaultdict(int)
    for li, cats in per_layer.items():
        for c, b in cats.items():
            comp[c] += b * (a.topk / a.experts) if c == 'moe_routed_bank' else b
    comp['lm_head'] = lm
    active = sum(comp.values())

    print(f'\n  active weight traffic per decode token: {active/2**20:.0f} MiB')
    print(f"  {'component':22s} {'MiB/token':>10s} {'share':>7s}")
    for c, b in sorted(comp.items(), key=lambda x: -x[1]):
        print(f'  {c:22s} {b/2**20:10.1f} {b/active*100:6.1f}%')

    print()
    for bw in (300, 350, 400):
        ms = active / (bw * 1e9) * 1000
        line = f'  at {bw} GB/s -> {ms:5.2f} ms/token = {1000/ms:6.1f} tok/s ceiling'
        if a.measured_ms:
            line += f'   ({a.measured_ms/ms:.1f}x headroom vs measured)'
        print(line)

if __name__ == '__main__':
    main()
