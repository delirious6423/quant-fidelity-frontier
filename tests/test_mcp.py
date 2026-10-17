#!/usr/bin/env python3
"""Black-box test of the QFF MCP server over real stdio: start it as a subprocess, list tools, call each one,
and check the answers against data.json computed independently here."""
import asyncio, json, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
SITE = os.environ.get("QFF_DATA", os.path.join(HERE, "_site"))
ok = fail = 0
def check(c, msg):
    global ok, fail
    c = bool(c); print(("[PASS] " if c else "[FAIL] ") + msg); ok += c; fail += (not c)
def J(res):
    if (getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)): s = getattr(res, "structured_content", None) or res.structuredContent; return s.get("result", s)
    return json.loads(res.content[0].text)
async def main():
    p = StdioServerParameters(command=sys.executable, args=[os.path.join(HERE, "qff_mcp", "qff_mcp.py")], env={**os.environ, "QFF_DATA": SITE})
    async with stdio_client(p) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = {t.name for t in (await s.list_tools()).tools}
            check(tools >= {"list_models", "search_quants", "get_repo", "list_sets", "frontier", "best_under"}, f"tools listed {sorted(tools)}")
            ms = J(await s.call_tool("list_models", {}))["models"]
            man = json.load(open(os.path.join(SITE, "models.json")))["models"]
            check([m["model"] for m in ms] == [m["model"] for m in man], f"list_models matches manifest ({len(ms)})")
            for M in man:
                d = json.load(open(os.path.join(SITE, M["data"]))); name = M["model"]
                _sz = sorted(r["size_gb"] for r in d["rows"] if r.get("size_gb")); BUD = round(_sz[len(_sz) // 2], 1)   # median size: a budget with plenty on both sides
                # search: every returned row exists verbatim in data.json, and the filter holds
                sr = J(await s.call_tool("search_quants", {"model": name, "fmt": "GGUF", "max_size_gb": BUD, "has_metric": "mean_kld", "limit": 200}))
                srcs = {(r["repo"], r["variant"], r.get("src")) for r in d["rows"]}
                check(sr["rows"] and all((x["repo"], x["variant"], x["src"]) in srcs and x["fmt"] == "GGUF" and x["size_gb"] <= BUD and "mean_kld" in x["claims"] for x in sr["rows"]),
                      f"{name}: search rows are real rows and respect filters ({sr['matches']})")
                ref = sum(1 for r in d["rows"] if r.get("fmt") == "GGUF" and r.get("size_gb") is not None and r["size_gb"] <= BUD and r.get("metrics", {}).get("mean_kld", {}).get("v") is not None)
                check(sr["matches"] == ref, f"{name}: search count matches independent count ({sr['matches']} vs {ref})")
                # raw strings are verbatim (no rounding or retyping)
                row = next(r for r in d["rows"] if r.get("metrics", {}).get("mean_kld", {}).get("v") is not None)
                g = J(await s.call_tool("get_repo", {"model": name, "repo": row["repo"]}))
                check(any(x["claims"].get("mean_kld") == row["metrics"]["mean_kld"]["raw"] for x in g["rows"]), f"{name}: get_repo returns the published string verbatim")
                # frontier equals an independent Pareto computation, for every comparable set
                bad = []
                for S in [x for x in d["sets"] if x.get("comparable")]:
                    k = (S.get("metrics") or [None])[0]
                    if not k: continue
                    fr = J(await s.call_tool("frontier", {"model": name, "set_id": S["id"]}))
                    P = [q for q in S["points"] if q.get("x") is not None and q.get("m", {}).get(k, {}).get("v") is not None and not q.get("is_ref")]
                    want = sorted((q["variant"], q["repo"]) for q in P if not any((o["x"] <= q["x"] and o["m"][k]["v"] < q["m"][k]["v"]) or (o["x"] < q["x"] and o["m"][k]["v"] <= q["m"][k]["v"]) for o in P if o is not q))
                    got = sorted((q["variant"], q["repo"]) for q in fr["efficient"])
                    if got != want or any(q.get("reference_build") for q in fr["efficient"]): bad.append(S["id"])
                check(not bad, f"{name}: frontier == independent Pareto on every comparable set {bad[:3]}")
                # not-comparable overview is refused for ranking
                res = await s.call_tool("frontier", {"model": name, "set_id": "all"})
                check((getattr(res, "is_error", None) or getattr(res, "isError", False)), f"{name}: frontier refuses the not-comparable overview")
                # best_under respects the budget and never mixes sets
                b = J(await s.call_tool("best_under", {"model": name, "max_size_gb": BUD}))
                check(b["per_set"] and all(e["best"]["x"] <= BUD for e in b["per_set"]) and len({e["set"] for e in b["per_set"]}) == len(b["per_set"]) and "own set" in b["caveat"],
                      f"{name}: best_under per set, within budget ({len(b['per_set'])} sets)")
            res = await s.call_tool("search_quants", {"model": "nonexistent-model"})
            check((getattr(res, "is_error", None) or getattr(res, "isError", False)), "unknown model is an error, not an empty answer")
asyncio.run(main()); print(f"{ok} passed, {fail} failed"); sys.exit(1 if fail else 0)
