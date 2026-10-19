#!/usr/bin/env python3
"""Column gate: a per-corpus value may sit under a corpus-only column header only when the caption right above the
table names that metric and no other (verify_columns.corpus_column_caption)."""
import ast, os, re, unittest
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(f"{CODE}/gates/verify_columns.py").read(); tree = ast.parse(src); ns = {"re": re}
want = {"FAMILY", "cells", "norm", "numtoks", "_strip_md", "table_span", "caption_above", "corpus_column_caption"}
for node in tree.body:
    if (isinstance(node, ast.FunctionDef) and node.name in want) or (isinstance(node, ast.Assign) and {getattr(t, "id", None) for t in node.targets} & want):
        exec(compile(ast.Module([node], []), "verify_columns.py", "exec"), ns)
ok = ns["corpus_column_caption"]; FAM = ns["FAMILY"]
KL, TOP, PPL = FAM[0][1], FAM[1][1], FAM[2][1]

def doc(caption):
    return [caption, "", "| | wikitext-2 | agentic text |", "|---|---|---|", "| **this W5** | **0.0424** | **0.0806** |"]

class Caption(unittest.TestCase):
    def check(self, caption, key="kld:wikitext-2", value="0.0424", header="wikitext-2", cell="**0.0424**", fam=KL):
        d = [ns["norm"](x) for x in doc(caption)]
        return ok(d, 4, header, ns["norm"](cell), key, value, fam)
    def test_metric_named_in_caption(self):
        self.assertTrue(self.check("Mean KL-divergence vs the bf16 model, full vocabulary, 2,048-token windows:"))
    def test_caption_without_metric_fails(self):
        self.assertFalse(self.check("Results on two corpora:"))
    def test_caption_naming_two_metrics_fails(self):
        self.assertFalse(self.check("Mean KL divergence and perplexity on two corpora:"))
    def test_wrong_corpus_column_fails(self):
        self.assertFalse(self.check("Mean KL divergence:", header="agentic text"))
    def test_not_a_per_corpus_key_fails(self):
        self.assertFalse(self.check("Mean KL divergence:", key="mean_kld"))
    def test_metric_in_caption_must_be_the_filed_one(self):
        self.assertFalse(self.check("Perplexity on two corpora:", key="ppl:wikitext-2", fam=KL))
        self.assertTrue(self.check("Perplexity on two corpora:", key="ppl:wikitext-2", fam=PPL))
    def test_cell_with_two_numbers_fails(self):
        self.assertFalse(self.check("Mean KL divergence:", cell="0.0424 ± 0.001"))
    def test_heading_is_not_a_caption(self):
        self.assertFalse(self.check("## KL divergence"))

if __name__ == "__main__": unittest.main()
