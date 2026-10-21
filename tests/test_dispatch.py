"""The hourly scan dispatches every mapping prompt it computes; max_parallel bounds concurrency, not the list.

Replays the actual dispatch branches of pipeline/qcl_next.py with inert stubs (pipeline/runner_baseline.py); no scan,
agent, network or git runs. On 2026-10-20 the eight-prompt slice left three models without a worker and cost a second
coordinator round of 6.5 minutes (logs/scan/20261020T1412Z-live.jsonl)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
from runner_baseline import dispatch_baseline


class Dispatch(unittest.TestCase):
    def test_every_mapping_prompt_starts_in_the_first_round(self):
        for models in (4, 12, 20):
            with self.subTest(models=models):
                result = dispatch_baseline(models)
                self.assertEqual(result['first_dispatch'], result['eligible_mapping_requests'])
                self.assertEqual(result['first_dispatch_unstarted'], 0)
                self.assertEqual(result['coordinator_handoffs'], 1)
                self.assertEqual(result['completed'], result['eligible_mapping_requests'])
                self.assertEqual(result['completed_repeated'], 0)


if __name__ == '__main__':
    unittest.main()
