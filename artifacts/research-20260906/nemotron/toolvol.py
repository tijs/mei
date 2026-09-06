"""Is the real benchmark failure driven by TOOL-SCHEMA volume, and does a
stronger instruction fix it?

Real failing rows were 21,934 prompt tokens. The Hermes system prompt alone is
4,739 and DOES produce a tool call; so ~17k of the real prompt is tool schemas.
Leg E earlier used 22 small tools (1,830 tokens) and worked. This scales tool
schema volume to the real level, then tries prompt-level fixes.
"""
import json, urllib.request, time
MODEL="mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit"
H=json.load(open('hermes_msgs.json')); USER=H['user']; HSYS=H['system']
BASE=[{"type":"function","function":{"name":"web_search","description":"Search the web for information.",
       "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
      {"type":"function","function":{"name":"write_file","description":"Write text to a file.",
       "parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]
DESC=("Performs a detailed operation on the target resource. This tool handles validation, "
      "normalisation, retry semantics and error reporting. Provide the arguments exactly as "
      "described; omitted optional arguments fall back to documented defaults. ")
def fat_tools(n):
    out=list(BASE)
    for i in range(n):
        props={f"arg_{j}":{"type":"string","description":DESC[:300]} for j in range(6)}
        out.append({"type":"function","function":{"name":f"resource_op_{i}","description":DESC*2,
                    "parameters":{"type":"object","properties":props,"required":[f"arg_0"]}}})
    return out
def call(label, sysm, tools):
    t0=time.time()
    r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8024/v1/chat/completions",
        data=json.dumps({"model":MODEL,"messages":[{"role":"system","content":sysm},{"role":"user","content":USER}],
                         "tools":tools,"max_tokens":300,"temperature":0.0}).encode(),
        headers={"Content-Type":"application/json"}), timeout=1200)
    d=json.loads(r.read()); dt=time.time()-t0
    u=d.get("usage",{}); m=d["choices"][0]["message"]
    pt=u.get("prompt_tokens",0); tc="YES" if m.get("tool_calls") else "no"
    print(f"  {label:50s} prompt={pt:6d} tool={tc:4s} finish={d['choices'][0].get('finish_reason')}")
    if tc=="no": print(f"      content: {repr((m.get('content') or '')[:170])}")
    return pt, tc
print("--- tool-schema volume, small system prompt ---")
for n in (10, 30, 60):
    call(f"{n+2} fat tools", "You are a helpful assistant.", fat_tools(n))
print("\n--- tool-schema volume + REAL hermes system prompt (matches benchmark) ---")
for n in (30, 60):
    call(f"hermes prompt + {n+2} fat tools", HSYS, fat_tools(n))
print("\n--- CANDIDATE FIXES at the failing condition (hermes prompt + 60 fat tools) ---")
NUDGE = HSYS + ("\n\nIMPORTANT: You have tools available. When the user's request requires "
                "information you do not have, or an action on the filesystem or network, you MUST "
                "call the appropriate tool. Do not answer from memory and do not merely describe "
                "what you intend to do — emit the tool call.")
call("fix A: explicit must-call-tools instruction", NUDGE, fat_tools(60))
