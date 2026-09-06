import json, urllib.request, time, sys
STEP=sys.argv[1] if len(sys.argv)>1 else "?"
MODEL="mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit"
UNIT="The following is background reference material for the assistant. "
def run(approx):
    sysm = UNIT * max(1, approx//11)
    t0=time.time()
    r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8024/v1/chat/completions",
        data=json.dumps({"model":MODEL,"messages":[{"role":"system","content":sysm},
                          {"role":"user","content":"Reply with exactly: ready"}],
                         "max_tokens":8,"temperature":0.0}).encode(),
        headers={"Content-Type":"application/json"}), timeout=1200)
    d=json.loads(r.read()); dt=time.time()-t0
    pt=d.get("usage",{}).get("prompt_tokens",0)
    return pt, dt, pt/dt if dt else 0
for target in (8000, 20000):
    for rep in range(2):
        pt,dt,pps = run(target)
        print(f"step={STEP} target={target} rep={rep+1} prompt={pt} wall={dt:.1f}s prefill={pps:.1f} pps")
