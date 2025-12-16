import json

with open('/home/csgrad/jmeem001/langgraph_scratch/dec16/output_base_gair/trace_out.json', 'r') as f:
    d1 = json.load(f)

with open('/home/csgrad/jmeem001/langgraph_scratch/dec16/output_temporal_verify/trace_out.json', 'r') as f:
    d2 = json.load(f)

def func(d):
    steps = 0
    for item in d:
        steps += len(item['trace'])
    
    return steps/len(d)

print(func(d1))
print(func(d2))
