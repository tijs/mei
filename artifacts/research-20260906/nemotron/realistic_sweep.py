"""Re-test the tool-call length threshold with REALISTIC varied filler.

The first sweep used one sentence repeated hundreds of times, which is
degenerate input a model may react to oddly -- and it conflicted with the
observation that the real 4,746-token Hermes prompt DID produce a tool call.
This uses varied natural prose+code drawn from real repository documentation.
"""
import json, urllib.request, time, glob, random
MODEL="mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit"
H=json.load(open('hermes_msgs.json')); USER=H['user']; HSYS=H['system']
TOOLS=[{"type":"function","function":{"name":"web_search","description":"Search the web for information.",
        "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
       {"type":"function","function":{"name":"write_file","description":"Write text to a file.",
        "parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]

# realistic varied corpus from the repo's own docs and source
corpus=[]
for pat in ("/Users/tijs/projects/local-model-bench/docs/*.md",
            "/Users/tijs/projects/local-model-bench/*.md",
            "/Users/tijs/projects/mei/docs/*.md",
            "/Users/tijs/projects/mei/*.md",
            "/Users/tijs/projects/local-model-bench/runner/*.py"):
    for f in glob.glob(pat):
        try: corpus.append(open(f, encoding='utf-8', errors='ignore').read())
        except Exception: pass
BIG="\n\n".join(corpus)
print(f"corpus chars available: {len(BIG)}")

def call(label, sysm, tools=TOOLS):
    t0=time.time()
    r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8024/v1/chat/completions",
        data=json.dumps({"model":MODEL,"messages":[{"role":"system","content":sysm},{"role":"user","content":USER}],
                         "tools":tools,"max_tokens":300,"temperature":0.0}).encode(),
        headers={"Content-Type":"application/json"}), timeout=1200)
    d=json.loads(r.read()); dt=time.time()-t0
    u=d.get("usage",{}); m=d["choices"][0]["message"]
    pt=u.get("prompt_tokens",0); tc="YES" if m.get("tool_calls") else "no"
    print(f"  {label:44s} prompt={pt:6d} {pt/dt:7.1f}pps tool={tc:4s} finish={d['choices'][0].get('finish_reason')}")
    return pt, tc

print(f"\n--- realistic varied filler (repo docs/source) ---")
for chars in (8000, 20000, 40000, 60000, 90000):
    call(f"varied prose {chars} chars", BIG[:chars])
print(f"\n--- REAL Hermes system prompt + varied filler appended ---")
for extra in (0, 20000, 60000):
    call(f"hermes prompt + {extra} chars", HSYS + ("\n\n" + BIG[:extra] if extra else ""))
print(f"\n--- degenerate repeated filler (the original, for contrast) ---")
UNIT="The following is background reference material for the assistant. "
for n in (200, 600):
    call(f"repeated sentence x{n}", UNIT*n)
