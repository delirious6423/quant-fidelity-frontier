#!/usr/bin/env python3
"""The content images a card embeds: charts and tables that may hold results or details the card text lacks.

Root-independent on purpose: pipeline/qcl_map.py re-exports these names, and pipeline/qcl_images.py (run from the
scripted runner on any model root, or none) imports them from here, so no caller has to load a model's snapshot index
just to find the image references on a card. Badges, logos and shields are ignored."""
import re

IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(\s*<?([^)\s>]+)[^)]*\)|<img\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"'][^>]*>", re.I)
IMAGE_SKIP = re.compile(r"shields\.io|badge|badgen|logo|avatar|emoji|star-history|visitor|favicon|icon|buymeacoffee|ko-fi|\.svg(\?|$)", re.I)


def card_images(lines):
    """[(line number, alt text, url)] for the content images on a card (badges and logos left out)."""
    found = []
    for n, line in enumerate(lines, 1):
        for m in IMAGE_RE.finditer(line):
            alt, url = (m.group(1), m.group(2)) if m.group(2) else ("", m.group(3))
            if url and not IMAGE_SKIP.search(url) and not IMAGE_SKIP.search(alt or ""): found.append((n, (alt or "").strip(), url))
    return found
