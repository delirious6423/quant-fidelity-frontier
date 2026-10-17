import json,sys
n=int(sys.argv[1]) if len(sys.argv)>1 else 45
cat=json.load(open("cache/catalogue.json"))["catalogue"]
man=json.load(open("cache/manifest.json"))["entries"]
have={k.rsplit("/README.md",1)[0] for k in man if k.endswith("/README.md")}
todo=[[c["id"],c["sha"]] for c in sorted(cat,key=lambda c:-(c["downloads"] or 0)) if f'{c["id"]}@{c["sha"]}' not in have]
print(len(todo),"missing",file=sys.stderr); print(json.dumps(todo[:n],separators=(",",":")))
