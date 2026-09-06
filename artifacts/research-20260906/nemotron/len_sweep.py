import json, urllib.request, time
exec(open('probe.py').read().split('Q="Find')[0])
H=json.load(open('hermes_msgs.json')); USER=H['user']
UNIT = "The following is background reference material for the assistant. "
def sysmsg(approx_tokens):
    # ~11 tokens per UNIT repetition
    return (UNIT * max(1, approx_tokens // 11))
print(f"{'target tok':>10s} {'actual':>8s} {'prefill pps':>12s} {'tool_call':>10s} {'finish':>12s}")
print("-"*60)
res=[]
for target in (2000, 6000, 10000, 14000, 16000, 18000, 22000):
    body_sys = sysmsg(target)
    t0=time.time()
    r=urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8024/v1/chat/completions",
        data=json.dumps({"model":MODEL,"messages":[{"role":"system","content":body_sys},{"role":"user","content":USER}],
                         "tools":TOOLS,"max_tokens":300,"temperature":0.0}).encode(),
        headers={"Content-Type":"application/json"}), timeout=900)
    d=json.loads(r.read()); dt=time.time()-t0
    u=d.get("usage",{}); m=d["choices"][0]["message"]
    pt=u.get("prompt_tokens",0); tc = "YES" if m.get("tool_calls") else "no"
    pps = pt/dt if dt else 0
    print(f"{target:10d} {pt:8d} {pps:12.1f} {tc:>10s} {str(d['choices'][0].get('finish_reason')):>12s}")
    res.append((pt,tc))
json.dump(res, open('bisect_results.json','w'))
