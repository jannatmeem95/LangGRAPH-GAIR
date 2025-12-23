import json

with open('/rhome/jmeem001/langgraph-scratch/dec16-LangGRAPH-GAIR/LangGRAPH-GAIR/cache/cached_urls_singlehop.json','r') as f:
    cached = json.load(f)


with open('/rhome/jmeem001/langgraph-scratch/dec16-LangGRAPH-GAIR/LangGRAPH-GAIR/cache/parallel_url_cache_all.json','r') as f:
    all = json.load(f)

x = {k:v for k,v in all.items() if k not in cached}

print(len(x))

with open('/rhome/jmeem001/langgraph-scratch/dec16-LangGRAPH-GAIR/LangGRAPH-GAIR/cache/parallel_url_cache_multihop.json', 'w') as f:
    json.dump(x,f,indent = 6)


"""with open('output/url_to_facts_multihop.json','r') as f:
    multi_facts = json.load(f)

with open('/rhome/jmeem001/langgraph-scratch/dec16-LangGRAPH-GAIR/LangGRAPH-GAIR/cache/url_to_facts.json','r') as f:
    single_facts = json.load(f) 

multi_facts.update(single_facts)"""

