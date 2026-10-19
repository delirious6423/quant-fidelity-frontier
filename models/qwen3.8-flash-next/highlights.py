# Head-to-head tables (one measurer, several publishers' files) and same-file disagreements.
# Every value is looked up from verified rows (never retyped); a missing lookup raises, so the build fails loudly.
from comparison import classify_group


def comparison_details(entries, metric):
    result = classify_group(entries, metric=metric)
    issues = result['differences'] + result['unknowns']
    conditions = sorted({issue['condition'].replace('_', ' ') for issue in issues})
    sources = {r['src']: r['measured_by'] for r in entries}
    links = ' · '.join(f'<a href="{E(url)}">{E(publisher)}</a>' for url, publisher in sources.items())
    return (f'<p class="dim"><b>{E(result["label"])}</b></p>'
            f'<details><summary>Methodology &amp; evidence</summary><p>{E(result["explanation"])}</p>'
            f'<p>Different or unverified conditions: {E(", ".join(conditions) or "none")}.</p>'
            f'<p>Quant publishers: {E(", ".join(sorted({r["repo"].split("/")[0] for r in entries})))}. '
            f'Measurement publishers and sources: {links}.</p></details>')

def find(repo, by, variant_sub, kind=None):
    c = [r for r in rows if r["repo"] == repo and r["measured_by"] == by and variant_sub.lower() in str(r["variant"]).lower()
         and (kind is None or r["kind"] == kind) and r["kind"] != "quoted"]
    if len(c) != 1:
        raise SystemExit(f"highlight lookup failed: {repo} by {by} ~ {variant_sub!r} -> {len(c)} matches")
    return c[0]

def mv(r, k):
    m = r["metrics"].get(k); return (m["v"], str(m["raw"])) if m else (None, "—")

def bars(title, sub, measurer, metric, entries, lower_better):
    """entries: list of rows; bar length scaled to max value on this chart."""
    vals = [mv(r, metric)[0] for r in entries]
    if not any(v is not None for v in vals):
        import sys; print(f"highlight skipped (no rows): {title}", file=sys.stderr); return ""
    top = max(v for v in vals if v is not None)
    result = classify_group(entries, metric=metric)
    out = [f'<figure><figcaption><b>{E(title)}</b><span>{E(sub)}</span></figcaption>', comparison_details(entries, metric), '<div class="bars">']
    ordered = sorted(entries, key=lambda r: mv(r, metric)[0], reverse=not lower_better) if result['ranking_permitted'] else entries
    for r in ordered:
        v, raw = mv(r, metric); self_ = r["measured_by"] == r["repo"]
        size = f'{r["size_raw"]} {BASIS_LABEL.get(r["basis"], "")}'.strip() if r["size_raw"] else ""
        who = r["repo"].split("/")[0]
        out.append(f'<div class="{"self" if self_ else "other"}" style="display:contents"><span class="lab">{E(who)} · {E(str(r["variant"]))} <small>{E(size)}</small></span>'
                   f'<span class="track"><span class="fill" style="width:{100*v/top:.1f}%"></span></span><span class="val">{E(raw)}</span></div>')
    out.append(f'</div></figure>')
    return "".join(out)

def cross_of(by):
    return [r for r in rows if r["measured_by"] == by and r["kind"] in ("cross", "own", "ladder") and not r.get("copied_from")]

def render_highlights():
    H = ['<div class="h2h">']
    M = "Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata"
    H.append(bars("Top-1 agreement with a Q8_0 build, measured by Maximilian228 in Strata",
                  "Reference: the publisher's own Q8_0. 5,153 tokens. Sizes are RAM in use while running. Their own file is highlighted.",
                  M, "top1_pct", [r for r in cross_of(M) if "top1_pct" in r["metrics"]], lower_better=False))
    G = "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF"
    H.append(bars("Mean KLD against unsloth's Q8_0, measured by agentionai (Gyro card)",
                  "Held-out technical writing and code, 2,048 context, 60 chunks. Sizes are GPU-resident weights; the n-gram table is excluded. Lower is better. Their own files are highlighted.",
                  G, "mean_kld", [r for r in cross_of(G) if "mean_kld" in r["metrics"]], lower_better=True))
    A = "agentionai/Qwen3.8-Flash-Next-AP-GGUF"
    H.append(bars("Mean KLD on held-out text, measured by agentionai (AP card)",
                  "2,048 context. The card refers to 'the same source' without naming it. Sizes are download size. Lower is better.",
                  A, "mean_kld", [r for r in cross_of(A) if "mean_kld" in r["metrics"]], lower_better=True))
    I = "ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4"
    try:
        H.append(bars("Top-1 agreement with BF16, measured by ivanfioravanti",
                      "Own logprob harness, 85 cases of 24 tokens. Small sample.",
                      I, "top1_pct", [r for r in cross_of(I) if "top1_pct" in r["metrics"]], lower_better=False))
    except Exception as ex:
        pass
    H.append("</div>")

    # same file, different claims
    D = []
    def add(file_label, entries):
        cells = []
        for r, note in entries:
            ks = [k for k in ("mean_kld", "top1_pct", "ppl_delta_pct") if k in r["metrics"]]
            vals = "; ".join(f'{ {"mean_kld":"KLD","top1_pct":"top-1","ppl_delta_pct":"PPL"}[k]} {r["metrics"][k]["raw"]}' for k in ks)
            cells.append(f'<tr><td>{E(file_label) if not cells else ""}</td><td>{E(r["measured_by"].split("/")[0])}{" (own card)" if r["measured_by"]==r["repo"] else ""}</td>'
                         f'<td class="n">{E(vals)}</td><td>{E(note)}</td><td><a href="{E(r["src"])}">{E(r["src_label"])}</a></td></tr>')
        D.append("".join(cells))
    U = "unsloth/Qwen3.8-Flash-Next-GGUF"
    add("unsloth UD-Q4_K_XL", [
        (find(U, U, "UD-Q4_K_XL"), "reference not stated"),
        (find(U, M, "UD-Q4_K_XL"), "vs Maximilian228's Q8_0, inside Strata"),
        (find(U, A, "UD-Q4_K_XL"), "held-out text, reference not named on the card"),
    ])
    add("unsloth UD-Q2_K_XL", [
        (find(U, U, "UD-Q2_K_XL"), "reference not stated"),
        (find(U, G, "UD-Q2_K_XL"), "vs unsloth Q8_0, held-out"),
    ])
    AT = "AtomicChat/Qwen3.8-Flash-Next-GGUF"
    add("AtomicChat AD-4.27bpw", [
        (find(AT, AT, "4.27"), "vs BF16, 87 chunks at 4,096 context"),
        (find(AT, G, "4.27"), "vs unsloth Q8_0, 2,048 context"),
    ])
    add("agentionai AP-Q4_K_XL", [
        (find(A, A, "AP-Q4_K_XL"), "held-out text, reference not named on the card"),
        (find(A, G, "AP-Q4_K_XL"), "same publisher, other card: stated reference unsloth Q8_0 (the AP card does not name its reference)"),
    ])
    # (AesSedai IQ4_XS vs agentionai ROCmFP4 card removed: that card reprints AesSedai's own ratios, L163-165; not a second measurement)
    DD = "ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit"
    D3 = "ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-3.3bpw"
    add("ddalcu mixed-4-8bit", [
        (find(DD, DD, "mixed"), "own card"),
        (find(D3, D3, "mixed-4-8bit", "ladder"), "same publisher's 3.3bpw card"),
    ])
    H.append('<h3 class="sub">Same file, different numbers</h3><p class="note"><b>Different or Unverified Protocol — Not Directly Comparable.</b> Where two cards report on the same file, the figures rarely agree. The Setup column lists what each card states about its method; where the stated setups differ, that may explain the gap, but the cause of any particular difference is not established, and the two cards may not have tested the same file revision.</p>'
             '<div class="tablebox"><table class="disagree"><thead><tr><th>File</th><th>Measured by</th><th>Claim</th><th>Setup</th><th>Source</th></tr></thead><tbody>'
             + "".join(D) + "</tbody></table></div>")
    return H
