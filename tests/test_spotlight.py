import json
import unittest
from pathlib import Path

import numpy as np

from experiment import WORDS, candidates, prompts
from spotlight.inverse import Unknown, forecast, recover, routing
from spotlight.oracle import Head

HERE = Path(__file__).parent


class InverseTests(unittest.TestCase):
    def test_matches_original_javascript_through_second_layer(self):
        ref = json.loads((HERE/'upstream_reference.json').read_text())
        ids = [WORDS.index(w) for w in ref['words']]
        for case in ref['targets']:
            box = Head(WORDS, case['layer'], case['head'])
            np.testing.assert_allclose(box.observe(ids), case['output'], atol=2e-13)
            np.testing.assert_allclose(box.reveal(ids), case['attention'], atol=2e-13)

    def test_recover_private_random_score_table_including_repeated_tokens(self):
        rng = np.random.default_rng(6021)
        table = rng.uniform(-2,2,(5,5,9))
        table[..., 0] = 0
        vectors = rng.normal(size=(5,4))
        def observe(ids):
            return routing(table, ids) @ vectors[list(ids)]
        vals, inferred, _ = recover(observe, 5, 8)
        np.testing.assert_allclose(inferred, table, atol=1e-10)
        np.testing.assert_allclose(vals, vectors, atol=1e-12)
        # Repeated identical words obscure their individual weights in this output.
        ids = [3]*9
        np.testing.assert_allclose(routing(inferred, ids), routing(table, ids), atol=1e-11)

    def test_unknown_if_values_collapse_or_become_collinear(self):
        for observe in [lambda ids: np.ones(4),
                        lambda ids: np.full(4, np.mean(ids))]:
            with self.assertRaises(Unknown):
                recover(observe, 4, 3)

    def test_position_only_control(self):
        rng = np.random.default_rng(4)
        vec = rng.normal(size=(4,3))
        def observe(ids):
            w = np.exp(-.2*np.arange(len(ids))[::-1])
            return (w/w.sum()) @ vec[list(ids)]
        _, table, _ = recover(observe, 4, 6)
        np.testing.assert_allclose(candidates(table)['lag'], table, atol=1e-12)

    def test_forecast_uses_measured_window(self):
        d = np.arange(24)
        table = np.random.default_rng(71).normal(size=(4,4,1))*np.sin(.65*d)
        future, _ = forecast(table[..., :17], 23)
        np.testing.assert_array_equal(future[..., :17], table[..., :17])
        np.testing.assert_allclose(future[..., 17:], table[..., 17:], atol=1e-9)

    def test_contextual_measurement_rejected(self):
        for h in [0,1]:
            with self.assertRaisesRegex(Unknown, 'outside measured value span'):
                recover(Head(WORDS, 1, h).observe, len(WORDS), 16)

    def test_heldout_rows_have_more_distinct_words_than_any_probe(self):
        pools = [set(prompts(s, 128, 12)) for s in [102,301,302,303]]
        self.assertEqual(len(set.union(*pools)), 4*128)
        self.assertTrue(all(len(set(x))>=4 for pool in pools for x in pool))


if __name__ == '__main__':
    unittest.main()
