import json, urllib.request
MODEL="mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit"
H=json.load(open('hermes_msgs.json')); USER=H['user']; HSYS=H['system']
DESC=("Performs a detailed operation on the target resource. This tool handles validation, "
      "normalisation, retry semantics and error reporting. ")
BASE=[{"type":"function","function":{"name":"web_search","description":"Search the web for information.",
       "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
      {"type":"function","function":{"name":"write_file","description":"Write text to a file.",
       "parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]
def fat(n):
    out=list(BASE)
    for i in range(n):
        out.append({"type":"function","function":{"name":f"resource_op_{i}","description":DESC*2,
            "parameters":{"type":"object","properties":{f"arg_{j}":{"type":"string","description":DESC[:300]} for j in range(6)},
                          "required":["arg_0"]}}})
    return out
def cap(label, sysm, tools, maxtok=400):
    r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8024/v1/chat/completions",
        data=json.dumps({"model":MODEL,"messages":[{"role":"system","content":sysm},{"role":"user","content":USER}],
                         "tools":tools,"max_tokens":maxtok,"temperature":0.0}).encode(),
        headers={"Content-Type":"application/json"}), timeout=1200)
    d=json.loads(r.read()); m=d["choices"][0]["message"]
    print("="*72); print(f"### {label}  prompt={d.get('usage',{}).get('prompt_tokens')} completion={d.get('usage',{}).get('completion_tokens')} finish={d['choices'][0].get('finish_reason')}")
    print(f"tool_calls: {'YES' if m.get('tool_calls') else 'NONE'}")
    print("--- FULL content ---"); print(m.get("content") or "<empty>")
    if m.get("reasoning_content"): print("--- reasoning ---"); print(m["reasoning_content"][:600])
cap("12 fat tools (emitted bare <function=>)", "You are a helpful assistant.", fat(10))
cap("hermes + 32 fat tools (real benchmark shape)", HSYS, fat(30))
