#!/usr/bin/env python3
"""README checks catch what they are for: a dead link, a renamed path, a missing live model (pipeline/readme_sync.py)."""
import ast, os, re, unittest
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tree = ast.parse(open(f"{CODE}/pipeline/readme_sync.py").read()); ns = {"os": os, "re": re, "CODE": CODE}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "dead_links": exec(compile(ast.Module([node], []), "readme_sync.py", "exec"), ns)
dead = ns["dead_links"]

class Readme(unittest.TestCase):
    def test_existing_links_and_paths_pass(self):
        self.assertEqual(dead("[m](docs/reference/METHOD.md) and `pipeline/qcl_live.py`, cards in `models/*/cache/cards/`"), [])
    def test_dead_link_and_renamed_path_are_caught(self):
        self.assertEqual(dead("[m](docs/NOPE.md) and `pipeline/gone.py`"), ["docs/NOPE.md", "pipeline/gone.py"])
    def test_urls_anchors_and_built_site_are_not_repo_paths(self):
        self.assertEqual(dead("[a](https://x.y/z) [b](#top) `/mcp` `../_site`"), [])
    def test_current_readme_is_consistent(self):
        import subprocess, sys
        p = subprocess.run([sys.executable, f"{CODE}/pipeline/readme_sync.py", "--check", "--structure"], capture_output=True, text=True)
        if "no built site" in p.stdout: self.skipTest("sites not built")
        self.assertEqual(p.returncode, 0, p.stdout)

if __name__ == "__main__": unittest.main()
