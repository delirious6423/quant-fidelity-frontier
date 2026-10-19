#!/usr/bin/env python3
"""Validate and assemble the complete static site, then replace the last good output.

  python3 assemble.py [OUT_DIR]        # default _site/

The output must be a new/empty directory or an existing QFF assembly. Model data,
repository sources and their ancestors are never valid output destinations.
"""
import html
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

CODE = Path(__file__).resolve().parent
BASE = "https://delirious6423.github.io/quant-fidelity-frontier/"
SITE_NAME = "QFF (Quant Fidelity Frontier)"
STATIC_SUFFIXES = {".html", ".json", ".txt", ".xml", ".css", ".js", ".svg", ".png", ".ico"}
PROTECTED_DIRS = {".git", ".github", "models", "gates", "pipeline", "web", "tests", "docs",
                  "protocol_evidence", "qff_mcp", "audit", "legacy", "logs"}
RESERVED_SLUGS = {"benchmarks", "guides"}
SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"


def read_json(path):
    with open(path, encoding="utf-8") as source:
        return json.load(source)


def valid_slug(slug):
    return (isinstance(slug, str) and re.fullmatch(r"[a-z0-9][a-z0-9.-]*", slug)
            and ".." not in slug and slug not in RESERVED_SLUGS)


def validate_output(output, code=CODE):
    """Refuse destructive targets before creating, copying or removing anything."""
    output, code = Path(output), Path(code).resolve()
    if output.is_symlink():
        raise ValueError("output must not be a symlink")
    target = output.resolve()
    if target == code or target in code.parents:
        raise ValueError("output must not be the repository or one of its ancestors")
    for name in PROTECTED_DIRS:
        source = (code / name).resolve()
        if target == source or source in target.parents:
            raise ValueError(f"output must not overlap repository sources: {name}")
    if target.exists():
        if not target.is_dir():
            raise ValueError("output must be a directory")
        if any(target.iterdir()):
            try:
                marker = read_json(target / "models.json")
                generated = ((target / ".nojekyll").is_file() and (target / "index.html").is_file()
                             and marker.get("name") == SITE_NAME and isinstance(marker.get("models"), list))
            except (OSError, ValueError, AttributeError):
                generated = False
            if not generated:
                raise ValueError("refusing to replace a nonempty directory that is not a QFF assembly")
    return target


def static_files(directory, skip_preview=False):
    """Validate every file, including hidden entries and symlinks, before copying."""
    result = []
    for path in sorted(Path(directory).rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlinks are not publishable: {path}")
        if path.is_dir():
            continue
        if skip_preview and path.relative_to(directory) == Path("artifact.html"):
            continue
        if not path.is_file() or (path.suffix not in STATIC_SUFFIXES and path.name != ".nojekyll"):
            raise ValueError(f"non-static file in site: {path}")
        if path.stat().st_size > 50_000_000:
            raise ValueError(f"file over 50 MB: {path}")
        result.append(path)
    return result


def preflight(code=CODE):
    """Every live root must have a complete, correctly identified model build."""
    code = Path(code).resolve()
    watchlist = read_json(code / "watchlist.json")
    if not isinstance(watchlist, dict) or not isinstance(watchlist.get("models"), list):
        raise ValueError("watchlist must contain a models list")
    status, order, checked, expected = {}, {}, {}, {}
    for item in watchlist["models"]:
        if not isinstance(item, dict) or not valid_slug(item.get("slug")):
            raise ValueError("invalid watchlist model slug")
        slug = item["slug"]
        if slug in status or item.get("status") not in {"live", "review", "ingesting", "queued"}:
            raise ValueError(f"duplicate slug or invalid status in watchlist: {slug}")
        status[slug], order[slug], checked[slug] = item["status"], len(order), item.get("last_scan_at")
        expected[slug] = item.get("model")
    configs = {path.parent.name: path for path in (code / "models").glob("*/model.json")}
    missing = sorted(slug for slug, state in status.items() if state == "live" and slug not in configs)
    if missing:
        raise ValueError(f"live models have no model.json: {missing}")
    models = []
    for folder, path in sorted(configs.items(), key=lambda item: (order.get(item[0], len(order)), item[0])):
        if status.get(folder, "live") != "live":
            continue
        config = read_json(path)
        if not isinstance(config, dict) or not valid_slug(config.get("slug")) or config["slug"] != folder:
            raise ValueError(f"model slug must match its directory: {path}")
        model = config.get("model")
        if not isinstance(model, str) or not model or (expected.get(folder) and expected[folder] != model):
            raise ValueError(f"model identity does not match watchlist: {path}")
        site = path.parent / "site"
        if path.is_symlink() or path.parent.is_symlink() or site.is_symlink() or not site.is_dir():
            raise ValueError(f"live model is missing a regular site directory: {folder}")
        files = static_files(site, skip_preview=True)
        for name in ("index.html", "data.json", "sitemap.xml"):
            if site / name not in files:
                raise ValueError(f"live model {folder} is missing {name}; run build.py first")
        data = read_json(site / "data.json")
        if not isinstance(data, dict) or data.get("model") != model:
            raise ValueError(f"built data model identity mismatch: {folder}")
        for key in ("repos", "rows", "sets"):
            if not isinstance(data.get(key), list) or any(not isinstance(item, dict) for item in data[key]):
                raise ValueError(f"built data {key} must be a list of objects: {folder}")
        ids = [item.get("id") for item in data["sets"]]
        if any(not isinstance(identifier, str) or not identifier for identifier in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"built data contains invalid or duplicate measurement-set ids: {folder}")
        sitemap = ET.parse(site / "sitemap.xml")
        if sitemap.getroot().tag != f"{{{SITEMAP_NS}}}urlset":
            raise ValueError(f"invalid model sitemap: {folder}")
        urls = [node.text for node in sitemap.iter(f"{{{SITEMAP_NS}}}loc")]
        if urls != [BASE + folder + "/"]:
            raise ValueError(f"model sitemap identity mismatch: {folder}")
        models.append({"site": site, "files": files, "sitemap": sitemap,
                       "manifest": {"model": model, "slug": folder, "data": folder + "/data.json",
                                    "captured_at": data.get("captured_at"), "checked_at": checked.get(folder),
                                    "repos": len(data["repos"]), "rows": len(data["rows"]), "sets": len(data["sets"])}})
    if not models:
        raise ValueError("refusing to assemble an empty site")
    return models


def render_site(out, models, code=CODE):
    manifest = [item["manifest"] for item in models]
    for item in models:
        for source in item["files"]:
            relative = source.relative_to(item["site"])
            if relative in (Path("sitemap.xml"), Path("robots.txt")):
                continue
            target = out / item["manifest"]["slug"] / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    ET.register_namespace("", SITEMAP_NS)
    sitemap = ET.Element(f"{{{SITEMAP_NS}}}urlset")
    home = ET.SubElement(sitemap, f"{{{SITEMAP_NS}}}url")
    ET.SubElement(home, f"{{{SITEMAP_NS}}}loc").text = BASE
    for item in models:
        sitemap.extend(item["sitemap"].getroot())
    ET.ElementTree(sitemap).write(out / "sitemap.xml", encoding="utf-8", xml_declaration=True)
    (out / "robots.txt").write_text(f"User-agent: *\nAllow: /\nSitemap: {BASE}sitemap.xml\n", encoding="utf-8")
    (out / "models.json").write_text(json.dumps({"name": SITE_NAME,
        "about": "Publisher claims about Hugging Face quants, each pinned to its source line. Claims, not measurements.",
        "models": manifest}, indent=1), encoding="utf-8")
    settings = read_json(Path(code) / "site.json") if (Path(code) / "site.json").exists() else {}
    analytics = settings.get("analytics") or {}
    script = (f'<script defer src="{html.escape(analytics.get("script") or "https://cloud.umami.is/script.js")}" data-website-id="{html.escape(analytics["website_id"])}" data-domains="delirious6423.github.io"></script>' if analytics.get("website_id") else "")
    support = (f' · <a href="{html.escape(settings["support_url"])}" rel="noopener" data-umami-event="support">{html.escape(settings.get("support_label") or "Support QFF")}</a>'
               f' (<a href="https://github.com/delirious6423/quant-fidelity-frontier/blob/main/docs/SPONSORS.md">sponsor policy</a>)') if settings.get("support_url") else ""
    from homepage import build_homepage
    from benchmark_view import build_benchmark_view
    from discovery import build_discovery
    build_benchmark_view(out, manifest, BASE, script, support)
    build_homepage(out, manifest, BASE, script, support)
    build_discovery(out, BASE, manifest, script)
    (out / ".nojekyll").touch()


def replace_output(staged, output):
    """Commit only a completed assembly; restore the old directory if rename fails."""
    backup = None
    if output.exists():
        backup = Path(tempfile.mkdtemp(prefix=f".{output.name}-previous-", dir=output.parent))
        backup.rmdir()
        os.replace(output, backup)
    try:
        os.replace(staged, output)
    except BaseException:
        if backup is not None:
            os.replace(backup, output)
        raise
    if backup is not None:
        try:
            shutil.rmtree(backup)
        except OSError as error:
            # The new output is already complete and committed. A cleanup
            # failure must not falsely report that publication itself failed.
            print(f"Assembled successfully; previous output remains at {backup}: {error}", file=sys.stderr)


def assemble(output, code=CODE):
    output = validate_output(output, code)
    models = preflight(code)
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent))
    try:
        render_site(staged, models, code)
        files = static_files(staged)
        total = sum(path.stat().st_size for path in files)
        if total >= 1_000_000_000:
            raise ValueError("published site exceeds the GitHub Pages 1 GB limit")
        replace_output(staged, output)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return {"out": str(output), "files": len(files), "MB": round(total / 1e6, 1)}


if __name__ == "__main__":
    try:
        print(json.dumps(assemble(sys.argv[1] if len(sys.argv) > 1 else CODE / "_site")))
    except (ValueError, OSError, ET.ParseError) as error:
        raise SystemExit(f"REFUSING TO ASSEMBLE: {error}")
