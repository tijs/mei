import json, urllib.request, time, sys
URL="http://127.0.0.1:8024/v1/chat/completions"
MODEL="mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit"
TOOLS=[{"type":"function","function":{"name":"web_search","description":"Search the web for information.",
        "parameters":{"type":"object","properties":{"query":{"type":"string","description":"search query"}},"required":["query"]}}},
       {"type":"function","function":{"name":"write_file","description":"Write text to a file.",
        "parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]
def call(name, msgs, tools=TOOLS, **extra):
    body={"model":MODEL,"messages":msgs,"max_tokens":600,"temperature":0.0}
    if tools: body["tools"]=tools
    body.update(extra)
    t0=time.time()
    r=urllib.request.urlopen(urllib.request.Request(URL,
        data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}), timeout=600)
    d=json.loads(r.read()); dt=time.time()-t0
    m=d["choices"][0]["message"]; u=d.get("usage",{})
    print(f"\n{'='*70}\n### {name}   ({dt:.1f}s, prompt={u.get('prompt_tokens')}, completion={u.get('completion_tokens')})")
    print(f"finish_reason: {d['choices'][0].get('finish_reason')}")
    print(f"tool_calls   : {json.dumps(m.get('tool_calls'))[:300] if m.get('tool_calls') else 'NONE'}")
    rc=m.get("reasoning_content")
    print(f"reasoning    : {repr(rc[:400]) if rc else 'NONE'}")
    print(f"content      : {repr((m.get('content') or '')[:500])}")
    return d
Q="Find the current population of Amsterdam and write it to /tmp/pop.txt"
call("A. minimal system + tools", [{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":Q}])
call("B. no system msg + tools",  [{"role":"user","content":Q}])
call("C. explicit tool nudge",    [{"role":"system","content":"You are a helpful assistant. Use the provided tools to accomplish the task."},{"role":"user","content":Q}])
