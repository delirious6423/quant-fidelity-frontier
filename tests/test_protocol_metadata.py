"""Build adapter regressions against actual pinned evidence and counterfactuals."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import protocol_metadata as adapter
from comparison import classify_pair


class EvidenceBindings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = ROOT / 'models/qwen3.8-27b'
        cls.idx = json.loads((cls.root / 'snapshots/index.json').read_text())
        cls.bindings = adapter.load_bindings(cls.root, cls.idx)
        cls.registry = json.loads((ROOT / 'protocol_evidence/qwen3.8-27b.json').read_text())

    def row(self, **overrides):
        row = dict(repo='AtomicChat/Qwen3.8-27B-GGUF', measured_by='AtomicChat/Qwen3.8-27B-GGUF',
                   sha='main', L=55, variant='fixture identity; no authored values', kind='own',
                   src='https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF/blob/main/README.md#L55',
                   weights_modified=False, metrics={}, methodology_override={})
        row.update(overrides)
        return row

    def attach(self, *rows):
        adapter.attach_protocols(rows, self.root, 'Qwen/Qwen3.8-27B', self.idx)
        return rows

    def test_exact_quotes_hashes_and_lines_validate(self):
        adapter._validate_sources(self.registry)
        self.assertEqual(len(self.bindings), 2)
        self.assertNotIn('issues', self.bindings['AtomicChat/Qwen3.8-27B-GGUF'][0]['protocol'])

    def test_source_hash_change_withdraws_evidence_without_removing_row(self):
        with patch.object(adapter, '_digest', return_value='changed'):
            a,b=self.attach(self.row(), self.row(variant='other'))
        result=classify_pair(a,b,'kld')
        self.assertFalse(result['ranking_permitted'])
        self.assertTrue(any(d['condition']=='source_binding' for d in result['differences']))
        self.assertEqual(a['repo'], 'AtomicChat/Qwen3.8-27B-GGUF')

    def test_tampered_quote_is_rejected(self):
        value=copy.deepcopy(self.registry)
        value['bindings'][0]['protocol']['base_model']['evidence'][0]['quote']='fabricated'
        with self.assertRaises(ValueError):adapter._validate_sources(value)

    def test_shared_evaluator_allows_cross_quant_publisher(self):
        a,b=self.attach(self.row(),self.row(repo='ggml-org (repo not named on card)',kind='cross'))
        self.assertTrue(classify_pair(a,b,'kld')['ranking_permitted'])
        self.assertEqual(a['measurement_owner'], b['measurement_owner'])
        self.assertNotEqual(a['quantization_owner'], b['quantization_owner'])
        self.assertNotEqual(a['measurement_id'], b['measurement_id'])

    def test_metric_evidence_does_not_leak(self):
        a,b=self.attach(self.row(),self.row(variant='other'))
        self.assertTrue(classify_pair(a,b,'top1')['ranking_permitted'])
        self.assertFalse(classify_pair(a,b,'ppl')['ranking_permitted'])

    def test_same_publisher_unregistered_card_is_not_evidence(self):
        a,b=self.attach(self.row(measured_by='AtomicChat/another-card'), self.row(measured_by='AtomicChat/another-card'))
        self.assertFalse(classify_pair(a,b,'kld')['ranking_permitted'])

    def test_row_protocol_override_is_not_swallowed(self):
        a,b=self.attach(self.row(),self.row(methodology_override={'ctx':'512'}))
        self.assertFalse(classify_pair(a,b,'kld')['ranking_permitted'])

    def test_copied_or_quoted_claim_does_not_become_independent(self):
        a,b=self.attach(self.row(),self.row(copied_from='other/source'))
        self.assertEqual(b['claim_provenance']['kind'],'copied')
        self.assertIsNone(b['measurement_owner'])
        self.assertEqual(b['claim_provenance']['claim_publisher'], b['measured_by'])
        self.assertFalse(classify_pair(a,b,'kld')['ranking_permitted'])
        a,b=self.attach(self.row(),self.row(kind='quoted'))
        self.assertIsNone(b['measurement_owner'])
        self.assertFalse(classify_pair(a,b,'kld')['ranking_permitted'])

    def test_conflicting_bindings_remain_contradictory(self):
        bindings=copy.deepcopy(self.bindings)
        entry=copy.deepcopy(bindings['AtomicChat/Qwen3.8-27B-GGUF'][0])
        entry['protocol']['context_length']['value']=512
        bindings['AtomicChat/Qwen3.8-27B-GGUF'].append(entry)
        with patch.object(adapter,'load_bindings',return_value=bindings):a,b=self.attach(self.row(),self.row())
        self.assertFalse(classify_pair(a,b,'kld')['ranking_permitted'])
        self.assertEqual(len(a['protocol']['context_length']),2)

    def test_existing_source_column_mapping_transfers_only_reviewed_overlay(self):
        a,b=self.row(),self.row()
        mapper=lambda row:{'kld':'mean_kld','ppl':'ppl:english'}
        adapter.attach_protocols([a,b],self.root,'Qwen/Qwen3.8-27B',self.idx,metric_mapper=mapper)
        self.assertTrue(classify_pair(a,b,'mean_kld')['ranking_permitted'])
        self.assertFalse(classify_pair(a,b,'ppl:english')['ranking_permitted'])


if __name__=='__main__':unittest.main()
