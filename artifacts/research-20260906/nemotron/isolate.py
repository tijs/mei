import json, urllib.request, time
exec(open('probe.py').read().split('Q="Find')[0])   # reuse call()/TOOLS
H=json.load(open('hermes_msgs.json'))
SYS, USER = H['system'], H['user']
FILLER = ("The following is background reference material for the assistant. " * 40 + "\n") * 45  # ~20k chars

many = []
for i in range(20):
    many.append({"type":"function","function":{"name":f"tool_{i}","description":f"Does operation {i} on a resource.",
        "parameters":{"type":"object","properties":{"arg":{"type":"string","description":"an argument"}},"required":["arg"]}}})
many_plus = many + TOOLS

call("D. REAL Hermes system prompt + 2 tools", [{"role":"system","content":SYS},{"role":"user","content":USER}])
call("E. small system + 22 tools",             [{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":USER}], tools=many_plus)
call("F. filler system (~20k chars) + 2 tools",[{"role":"system","content":FILLER},{"role":"user","content":USER}])
