import unittest

from load_rerank import select_load_rerank_action


class LoadRerankTests(unittest.TestCase):
    def test_zero_weight_keeps_qatten_argmax(self):
        action, info = select_load_rerank_action(
            q_values=[1.0, 1.2, 1.1],
            load_penalty=[0.0, 1.0, 0.0],
            mode="margin_gated",
            margin_threshold=1.0,
            penalty_weight=0.0,
        )

        self.assertEqual(action, 1)
        self.assertFalse(info["action_changed"])

    def test_margin_candidate_outside_cannot_be_selected(self):
        action, info = select_load_rerank_action(
            q_values=[1.0, 0.95, 0.1],
            load_penalty=[1.0, 1.0, 0.0],
            mode="margin_gated",
            margin_threshold=0.1,
            penalty_weight=10.0,
        )

        self.assertIn(action, [0, 1])
        self.assertNotIn(2, info["candidate_actions"])

    def test_high_load_penalty_demotes_candidate(self):
        action, info = select_load_rerank_action(
            q_values=[1.0, 0.95, 0.2],
            load_penalty=[1.0, 0.0, 0.0],
            mode="margin_gated",
            margin_threshold=0.1,
            penalty_weight=0.2,
        )

        self.assertEqual(action, 1)
        self.assertTrue(info["action_changed"])

    def test_topk_candidate_outside_cannot_be_selected(self):
        action, info = select_load_rerank_action(
            q_values=[1.0, 0.99, 0.98, 0.1],
            load_penalty=[1.0, 0.0, 0.0, 0.0],
            mode="topk_rerank",
            topk=2,
            penalty_weight=10.0,
        )

        self.assertEqual(action, 1)
        self.assertEqual(info["candidate_actions"], [0, 1])

    def test_agent1_is_not_reranked(self):
        action, info = select_load_rerank_action(
            q_values=[1.0, 1.1],
            load_penalty=[0.0, 10.0],
            agent_num=1,
            mode="margin_gated",
            margin_threshold=1.0,
            penalty_weight=10.0,
        )

        self.assertEqual(action, 1)
        self.assertFalse(info["applied"])


if __name__ == "__main__":
    unittest.main()
