---
title: QFF MCP
emoji: 📏
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---
Read-only MCP server for QFF (Quant Fidelity Frontier): publisher claims about Hugging Face quants, each pinned to its
card line. Connector URL: `https://<this-space>.hf.space/mcp`. Copy `../qff_mcp.py` and the repository-root `comparison.py` next to the Dockerfile before pushing.
The same classifier runs in the static site builder and MCP server; without it rankings fail closed.
The nine read-only tools include reviewed benchmark discovery and cross-model task-score search.
The Space's runtime `SPACE_HOST` automatically configures the exact public Host and HTTPS Origin allowlists;
DNS-rebinding protection remains enabled. For another hosting origin, set `QFF_PUBLIC_URL` to its exact HTTPS URL.
The Docker recipe pins the SDK version tested by CI. No extra benchmark module needs to be copied.
From the repository root, prepare a separate Space checkout with:

```bash
cp qff_mcp/qff_mcp.py comparison.py /path/to/space-checkout/
cp qff_mcp/space/Dockerfile qff_mcp/space/README.md /path/to/space-checkout/
```

No hosting changes are needed for the static GitHub Pages site.
Publishing new site data refreshes MCP's data cache within five minutes of the next request.
New tools and server fixes require updating this Space's copied Python files and rebuilding the container.
