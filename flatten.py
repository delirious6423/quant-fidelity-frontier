import json,glob,os,sys
out={}
for f in sorted(glob.glob("claims/*.json")):
    d=json.load(open(f)); r=d.get("repo") or os.path.basename(f)
    items=[]
    def walk(x,path):
        if isinstance(x,bool) or x is None: return
        if isinstance(x,(int,float)): items.append({"path":path,"value":x})
        elif isinstance(x,dict):
            for k,v in x.items(): walk(v,f"{path}.{k}")
        elif isinstance(x,list):
            for i,v in enumerate(x): walk(v,f"{path}[{v.get('name',i) if isinstance(v,dict) else i}]")
    for key in ["variants","baseline_numbers","methodology"]:
        walk(d.get(key),key)
    out[r]={"evidence":d.get("evidence_urls",[]),"nums":items}
json.dump(out,open("flat.json","w"))
print(sum(len(v["nums"]) for v in out.values()),"numbers across",len(out),"files")
