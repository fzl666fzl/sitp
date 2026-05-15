import unittest
import sys
import tempfile
import os

import numpy as np
import simpy
import torch
import random

sys.argv = sys.argv[:1]

from agent import Agents
from config_gnn_qatten_end2end import Config
from GNN_QATTEN_END2END_dis1 import (
    end2end_best_score,
    is_better_online_balanced_summary,
    is_better_online_pulse_summary,
    is_better_end2end_summary,
    load_saved_online_balanced_score,
    load_best_existing_online_balanced_score,
    load_saved_end2end_score,
    load_saved_online_pulse_score,
    online_balanced_score,
    best_existing_online_balanced_tag,
    copy_tagged_model,
    save_tagged_model,
)
from GNN_QATTEN_END2END_online import (
    configure_online_defaults,
    online_init_pulse,
    set_eval_seed as set_online_seed,
)
from GNN_QATTEN_END2END_eval_checkpoints import eval_tags
import rollout_dis as base
import rollout_end2end as end2end


class End2EndActionSpaceTest(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        for proc_idx, proc_id in enumerate(end2end.pro_id):
            for team_id in range(end2end.team_num):
                action_id = end2end.encode_combo_action(proc_idx, team_id)
                decoded_proc_idx, decoded_proc_id, decoded_team_id = end2end.decode_combo_action(action_id)
                self.assertEqual(decoded_proc_idx, proc_idx)
                self.assertEqual(decoded_proc_id, int(proc_id))
                self.assertEqual(decoded_team_id, team_id)
                self.assertGreaterEqual(action_id, 0)
                self.assertLess(action_id, end2end.pro_num * end2end.team_num)

    def test_initial_mask_contains_only_initial_ready_orders(self):
        env = simpy.Environment()
        allstation, _, air = base.reset_env(env, 750)
        station = allstation[0]
        planned_team_finish = [0.0 for _ in range(end2end.team_num)]
        mask = end2end.build_end2end_avail_mask(
            air, station, set(), planned_team_finish, 750
        )
        valid_actions = np.nonzero(mask > 0)[0].tolist()
        valid_proc_ids = {
            end2end.decode_combo_action(action_id)[1] for action_id in valid_actions
        }
        self.assertEqual(valid_proc_ids, set(end2end.get_available_combo_orders(air)))

    def test_finished_and_planned_orders_are_masked(self):
        env = simpy.Environment()
        allstation, _, air = base.reset_env(env, 750)
        station = allstation[0]
        air.isfinish[1] = 1
        air.order_finish.append(1)
        planned_orders = {2}
        planned_team_finish = [0.0 for _ in range(end2end.team_num)]
        mask = end2end.build_end2end_avail_mask(
            air, station, planned_orders, planned_team_finish, 750
        )
        for team_id in range(end2end.team_num):
            self.assertEqual(mask[end2end.encode_combo_action(0, team_id)], 0.0)
            self.assertEqual(mask[end2end.encode_combo_action(1, team_id)], 0.0)

    def test_agent_does_not_choose_masked_action(self):
        conf = Config()
        conf.load_model = False
        agents = Agents(conf)
        agents.policy.init_hidden(1)
        state = np.zeros(conf.state_shape).tolist()
        last_action = np.zeros(conf.n_actions)
        avail_mask = np.zeros(conf.n_actions)
        valid_action = 7
        avail_mask[valid_action] = 1.0
        for _ in range(5):
            action = agents.choose_action(
                state, last_action, 0, avail_mask, epsilon=1.0, evaluate=False
            )
            self.assertEqual(action, valid_action)

    def test_rollout_shapes(self):
        conf = Config()
        conf.load_model = False
        conf.episode_limit = 60
        agents = Agents(conf)
        episode, _, _, summary = end2end.generate_episode(
            agents, conf, [], 750, 0, [], evaluate=True
        )
        self.assertEqual(episode["u"].shape, (1, conf.episode_limit, 1, 1))
        self.assertEqual(episode["u_onehot"].shape, (1, conf.episode_limit, 1, conf.n_actions))
        self.assertEqual(episode["avail_u"].shape, (1, conf.episode_limit, 1, conf.n_actions))
        self.assertEqual(
            episode["combo_pair_features"].shape,
            (1, conf.episode_limit, conf.n_actions, conf.combo_pair_feature_dim),
        )
        self.assertEqual(
            episode["combo_pair_features_"].shape,
            (1, conf.episode_limit, conf.n_actions, conf.combo_pair_feature_dim),
        )
        np.testing.assert_array_equal(
            episode["combo_pair_features"][0, 0, :, 0] > 0,
            episode["avail_u"][0, 0, 0, :] > 0,
        )
        self.assertEqual(summary["finished_order_count"], end2end.pro_num)
        trace_proc_ids = [item["proc_id"] for item in summary["actions"]]
        self.assertEqual(len(trace_proc_ids), end2end.pro_num)
        self.assertEqual(set(trace_proc_ids), set(int(item) for item in end2end.pro_id))
        self.assertEqual(len(trace_proc_ids), len(set(trace_proc_ids)))
        for item in summary["actions"]:
            _, proc_id, team_id = end2end.decode_combo_action(item["action_id"])
            self.assertEqual(proc_id, item["proc_id"])
            self.assertEqual(team_id, item["team_id"])

    def test_combo_action_scorer_shape(self):
        conf = Config()
        conf.load_model = False
        agents = Agents(conf)
        states = torch.zeros(2, 3, conf.state_shape, device=agents.policy.device)
        features = torch.zeros(
            2,
            3,
            conf.n_actions,
            conf.combo_pair_feature_dim,
            device=agents.policy.device,
        )
        with torch.no_grad():
            q_values = agents.policy.eval_combo_scorer(
                agents.policy.eval_graph_encoder.node_embeddings(),
                states,
                features,
                agents.policy.combo_action_proc_indices,
            )
        self.assertEqual(q_values.shape, (2, 3, 1, conf.n_actions))

    def test_teacher_init_pulse_infers_from_episode_times(self):
        summary = {
            "actions": [
                {"action_id": 1, "episode_time": 0.0, "disturbance": 0.5},
                {"action_id": 2, "episode_time": 570.0, "disturbance": 0.0},
                {"action_id": 3, "episode_time": 1140.0, "disturbance": 1.5},
            ],
            "final_pulse": 585.0,
        }
        self.assertEqual(end2end.infer_teacher_init_pulse(summary, 608), 570)
        self.assertEqual(end2end.extract_action_ids_from_summary(summary), [1, 2, 3])
        self.assertEqual(end2end.extract_disturbances_from_summary(summary), [0.5, 0.0, 1.5])

    def test_forced_action_trace_replays_same_actions(self):
        conf = Config()
        conf.load_model = False
        agents = Agents(conf)
        random.seed(133)
        np.random.seed(133)
        torch.manual_seed(133)
        _, _, _, summary = end2end.generate_episode(
            agents, conf, [], 608, 0, [], evaluate=True
        )
        action_ids = end2end.extract_action_ids_from_summary(summary)

        random.seed(133)
        np.random.seed(133)
        torch.manual_seed(133)
        _, _, _, replay_summary = end2end.generate_episode(
            agents,
            conf,
            [],
            608,
            0,
            [],
            evaluate=True,
            forced_action_trace=action_ids,
            strict_forced_trace=True,
            disable_disturbance=False,
        )
        self.assertEqual(
            end2end.extract_action_ids_from_summary(replay_summary),
            action_ids,
        )
        self.assertTrue(replay_summary["teacher_forcing"])
        self.assertEqual(replay_summary["forced_action_trace_consumed"], len(action_ids))

    def test_best_score_only_updates_finished_better_summary(self):
        finished = {
            "final_pulse": 600.0,
            "smoothness_index": 5.0,
            "finished_order_count": end2end.pro_num,
        }
        unfinished = {
            "final_pulse": 500.0,
            "smoothness_index": 1.0,
            "finished_order_count": end2end.pro_num - 1,
        }
        worse = {
            "final_pulse": 610.0,
            "smoothness_index": 10.0,
            "finished_order_count": end2end.pro_num,
        }
        self.assertEqual(end2end_best_score(finished), 605.0)
        self.assertIsNone(end2end_best_score(unfinished))
        should_update, score = is_better_end2end_summary(finished, None)
        self.assertTrue(should_update)
        self.assertEqual(score, 605.0)
        should_update, _ = is_better_end2end_summary(unfinished, 605.0)
        self.assertFalse(should_update)
        should_update, _ = is_better_end2end_summary(worse, 605.0)
        self.assertFalse(should_update)

    def test_weighted_best_score_uses_si_weight(self):
        summary = {
            "final_pulse": 600.0,
            "smoothness_index": 5.0,
            "finished_order_count": end2end.pro_num,
        }
        self.assertEqual(
            end2end_best_score(summary, "pulse_plus_weighted_si", si_weight=5.0),
            625.0,
        )
        should_update, score = is_better_end2end_summary(
            summary, 626.0, "pulse_plus_weighted_si", si_weight=5.0
        )
        self.assertTrue(should_update)
        self.assertEqual(score, 625.0)

    def test_online_pulse_best_prefers_lower_pulse_then_si(self):
        current_best = (600.0, 20.0)
        lower_pulse = {
            "final_pulse": 599.0,
            "smoothness_index": 100.0,
            "finished_order_count": end2end.pro_num,
        }
        same_pulse_lower_si = {
            "final_pulse": 600.0,
            "smoothness_index": 19.0,
            "finished_order_count": end2end.pro_num,
        }
        same_pulse_higher_si = {
            "final_pulse": 600.0,
            "smoothness_index": 21.0,
            "finished_order_count": end2end.pro_num,
        }
        unfinished = {
            "final_pulse": 500.0,
            "smoothness_index": 1.0,
            "finished_order_count": end2end.pro_num - 1,
        }

        self.assertEqual(is_better_online_pulse_summary(lower_pulse, current_best), (True, (599.0, 100.0)))
        self.assertEqual(is_better_online_pulse_summary(same_pulse_lower_si, current_best), (True, (600.0, 19.0)))
        self.assertEqual(is_better_online_pulse_summary(same_pulse_higher_si, current_best), (False, (600.0, 21.0)))
        self.assertEqual(is_better_online_pulse_summary(unfinished, current_best), (False, None))

    def test_online_balanced_best_uses_pulse_plus_weighted_si(self):
        current_best = 605.0
        lower_pulse_but_worse_si = {
            "final_pulse": 599.0,
            "smoothness_index": 20.0,
            "finished_order_count": end2end.pro_num,
        }
        balanced_better = {
            "final_pulse": 601.0,
            "smoothness_index": 3.0,
            "finished_order_count": end2end.pro_num,
        }
        unfinished = {
            "final_pulse": 500.0,
            "smoothness_index": 1.0,
            "finished_order_count": end2end.pro_num - 1,
        }

        self.assertEqual(online_balanced_score(balanced_better, si_weight=1.0), 604.0)
        self.assertEqual(
            is_better_online_balanced_summary(
                lower_pulse_but_worse_si, current_best, si_weight=1.0
            ),
            (False, 619.0),
        )
        self.assertEqual(
            is_better_online_balanced_summary(
                balanced_better, current_best, si_weight=1.0
            ),
            (True, 604.0),
        )
        self.assertEqual(
            is_better_online_balanced_summary(unfinished, current_best, si_weight=1.0),
            (False, None),
        )

    def test_saved_best_scores_can_initialize_training_baselines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            conf = Config()
            conf.model_dir = tmpdir
            conf.load_model = False
            agents = Agents(conf)
            summary = {
                "final_pulse": 600.0,
                "smoothness_index": 5.0,
                "finished_order_count": end2end.pro_num,
            }
            save_tagged_model(agents, conf, summary, 605.0, "validation_best_reproduce")
            save_tagged_model(agents, conf, summary, 600.0, "online_pulse_best")
            save_tagged_model(agents, conf, summary, 605.0, "online_balanced_best")

            self.assertEqual(
                load_saved_end2end_score(
                    agents,
                    "validation_best_reproduce",
                    "pulse_plus_weighted_si",
                    5.0,
                ),
                625.0,
            )
            self.assertEqual(
                load_saved_online_pulse_score(agents, "online_pulse_best"),
                (600.0, 5.0),
            )
            self.assertEqual(
                load_saved_online_balanced_score(
                    agents, "online_balanced_best", si_weight=1.0
                ),
                605.0,
            )
            self.assertEqual(
                load_best_existing_online_balanced_score(
                    agents,
                    ["online_balanced_best", "validation_best_reproduce", "online_pulse_best"],
                    si_weight=1.0,
                ),
                605.0,
            )
            source_tag, source_score = best_existing_online_balanced_tag(
                agents,
                ["online_balanced_best", "validation_best_reproduce", "online_pulse_best"],
                si_weight=1.0,
            )
            self.assertEqual(source_tag, "online_balanced_best")
            self.assertEqual(source_score, 605.0)

    def test_copy_tagged_model_syncs_online_balanced_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            conf = Config()
            conf.model_dir = tmpdir
            conf.load_model = False
            agents = Agents(conf)
            summary = {
                "final_pulse": 585.0,
                "smoothness_index": 8.0,
                "finished_order_count": end2end.pro_num,
            }
            save_tagged_model(agents, conf, summary, 593.0, "validation_best")
            copy_tagged_model(agents, "validation_best", "online_balanced_best")

            load_conf = Config()
            load_conf.model_dir = tmpdir
            load_conf.load_model = True
            load_conf.model_tag = "online_balanced_best"
            Agents(load_conf)
            self.assertEqual(load_conf.loaded_model_tag, "online_balanced_best")
            self.assertIsNotNone(load_conf.loaded_combo_scorer_path)
            self.assertEqual(
                load_saved_online_balanced_score(
                    agents, "online_balanced_best", si_weight=1.0
                ),
                593.0,
            )

    def test_online_script_uses_env_seed_and_init_pulse(self):
        old_seed = os.environ.get("END2END_ONLINE_SEED")
        old_init_pulse = os.environ.get("END2END_ONLINE_INIT_PULSE")
        try:
            os.environ["END2END_ONLINE_SEED"] = "321"
            os.environ["END2END_ONLINE_INIT_PULSE"] = "599"
            conf = Config()
            set_online_seed(conf)
            self.assertEqual(conf.seed, 321)
            self.assertEqual(online_init_pulse(), 599)
        finally:
            if old_seed is None:
                os.environ.pop("END2END_ONLINE_SEED", None)
            else:
                os.environ["END2END_ONLINE_SEED"] = old_seed
            if old_init_pulse is None:
                os.environ.pop("END2END_ONLINE_INIT_PULSE", None)
            else:
                os.environ["END2END_ONLINE_INIT_PULSE"] = old_init_pulse

    def test_online_defaults_match_validation_context(self):
        old_model_tag = os.environ.get("END2END_MODEL_TAG")
        old_model_dir = os.environ.get("END2END_MODEL_DIR")
        old_online_seed = os.environ.get("END2END_ONLINE_SEED")
        old_online_init_pulse = os.environ.get("END2END_ONLINE_INIT_PULSE")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.environ["END2END_MODEL_DIR"] = tmpdir
                os.environ.pop("END2END_MODEL_TAG", None)
                os.environ.pop("END2END_ONLINE_SEED", None)
                os.environ.pop("END2END_ONLINE_INIT_PULSE", None)

                save_conf = Config()
                save_conf.model_dir = tmpdir
                save_conf.load_model = False
                agents = Agents(save_conf)
                summary = {
                    "final_pulse": 600.0,
                    "smoothness_index": 5.0,
                    "finished_order_count": end2end.pro_num,
                }
                save_tagged_model(agents, save_conf, summary, 605.0, "validation_best")

                conf = configure_online_defaults(Config())
                self.assertTrue(conf.load_model)
                self.assertEqual(conf.model_tag, conf.end2end_validation_best_tag)

                worse_balanced_summary = {
                    "final_pulse": 606.0,
                    "smoothness_index": 4.0,
                    "finished_order_count": end2end.pro_num,
                }
                save_tagged_model(
                    agents, save_conf, worse_balanced_summary, 610.0, "online_balanced_best"
                )
                conf = configure_online_defaults(Config())
                self.assertEqual(conf.model_tag, conf.end2end_validation_best_tag)

                better_balanced_summary = {
                    "final_pulse": 590.0,
                    "smoothness_index": 4.0,
                    "finished_order_count": end2end.pro_num,
                }
                save_tagged_model(
                    agents, save_conf, better_balanced_summary, 594.0, "online_balanced_best"
                )
                conf = configure_online_defaults(Config())
                self.assertEqual(conf.model_tag, conf.end2end_online_balanced_best_tag)

                set_online_seed(conf)
                self.assertEqual(conf.seed, conf.end2end_validation_seed)
                self.assertEqual(online_init_pulse(conf), conf.end2end_validation_init_pulse)
        finally:
            if old_model_tag is None:
                os.environ.pop("END2END_MODEL_TAG", None)
            else:
                os.environ["END2END_MODEL_TAG"] = old_model_tag
            if old_model_dir is None:
                os.environ.pop("END2END_MODEL_DIR", None)
            else:
                os.environ["END2END_MODEL_DIR"] = old_model_dir
            if old_online_seed is None:
                os.environ.pop("END2END_ONLINE_SEED", None)
            else:
                os.environ["END2END_ONLINE_SEED"] = old_online_seed
            if old_online_init_pulse is None:
                os.environ.pop("END2END_ONLINE_INIT_PULSE", None)
            else:
                os.environ["END2END_ONLINE_INIT_PULSE"] = old_online_init_pulse

    def test_end2end_main_result_defaults(self):
        old_final_penalty = os.environ.get("END2END_FINAL_SI_PENALTY_WEIGHT")
        old_si_weight = os.environ.get("END2END_BEST_SI_WEIGHT")
        old_score_mode = os.environ.get("END2END_BEST_SCORE_MODE")
        try:
            os.environ.pop("END2END_FINAL_SI_PENALTY_WEIGHT", None)
            os.environ.pop("END2END_BEST_SI_WEIGHT", None)
            os.environ.pop("END2END_BEST_SCORE_MODE", None)
            conf = Config()
            self.assertEqual(conf.end2end_best_score_mode, "pulse_plus_si")
            self.assertEqual(conf.end2end_best_si_weight, 1.0)
            self.assertEqual(conf.end2end_final_si_penalty_weight, 0.0)
        finally:
            if old_final_penalty is None:
                os.environ.pop("END2END_FINAL_SI_PENALTY_WEIGHT", None)
            else:
                os.environ["END2END_FINAL_SI_PENALTY_WEIGHT"] = old_final_penalty
            if old_si_weight is None:
                os.environ.pop("END2END_BEST_SI_WEIGHT", None)
            else:
                os.environ["END2END_BEST_SI_WEIGHT"] = old_si_weight
            if old_score_mode is None:
                os.environ.pop("END2END_BEST_SCORE_MODE", None)
            else:
                os.environ["END2END_BEST_SCORE_MODE"] = old_score_mode

    def test_eval_checkpoints_includes_online_and_reproduce_tags(self):
        tags = eval_tags()
        self.assertIn("expert_trace_best", tags)
        self.assertIn("online_balanced_best", tags)
        self.assertIn("online_pulse_best", tags)
        self.assertLess(tags.index("expert_trace_best"), tags.index("latest"))
        self.assertLess(tags.index("online_balanced_best"), tags.index("latest"))
        self.assertLess(tags.index("online_pulse_best"), tags.index("latest"))

    def test_training_epsilon_decays_to_zero_before_late_epochs(self):
        conf = Config()
        conf.start_epsilon = 0.5
        conf.n_epochs = 100
        conf.end2end_epsilon_decay_fraction = 0.70
        self.assertEqual(end2end.episode_epsilon(conf, 0, evaluate=True), 0)
        self.assertEqual(end2end.episode_epsilon(conf, 0, evaluate=False), 0.5)
        self.assertAlmostEqual(end2end.episode_epsilon(conf, 35, evaluate=False), 0.25)
        self.assertEqual(end2end.episode_epsilon(conf, 70, evaluate=False), 0)
        self.assertEqual(end2end.episode_epsilon(conf, 99, evaluate=False), 0)

    def test_last_station_overload_penalty_lowers_last_station_reward(self):
        conf = Config()
        conf.end2end_last_station_penalty_weight = 1.0
        action_trace = [
            {"station_id": 0},
            {"station_id": end2end.station_num - 1},
        ]
        rewards, info = end2end.calculate_end2end_final_rewards(
            conf,
            1600.0,
            0.0,
            [600.0, 600.0, 600.0, 1600.0],
            action_trace,
            600.0,
        )
        self.assertGreater(info["last_station_overload"], 0.0)
        self.assertLess(rewards[1][0], rewards[0][0])

    def test_final_si_penalty_lowers_high_si_reward(self):
        conf = Config()
        conf.end2end_last_station_penalty_weight = 0.0
        conf.end2end_final_si_penalty_weight = 10.0
        action_trace = [{"station_id": 0}]
        low_si_rewards, low_info = end2end.calculate_end2end_final_rewards(
            conf,
            600.0,
            25.0,
            [590.0, 595.0, 600.0, 598.0],
            action_trace,
            600.0,
        )
        high_si_rewards, high_info = end2end.calculate_end2end_final_rewards(
            conf,
            600.0,
            400.0,
            [560.0, 580.0, 600.0, 620.0],
            action_trace,
            600.0,
        )
        self.assertGreater(high_info["final_si_penalty"], low_info["final_si_penalty"])
        self.assertLess(high_si_rewards[0][0], low_si_rewards[0][0])

    def test_validation_best_checkpoint_loads_combo_scorer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            conf = Config()
            conf.model_dir = tmpdir
            conf.load_model = False
            agents = Agents(conf)
            summary = {
                "final_pulse": 600.0,
                "smoothness_index": 5.0,
                "finished_order_count": end2end.pro_num,
            }
            save_tagged_model(agents, conf, summary, 605.0, "validation_best")

            load_conf = Config()
            load_conf.model_dir = tmpdir
            load_conf.load_model = True
            load_conf.model_tag = "validation_best"
            Agents(load_conf)
            self.assertEqual(load_conf.loaded_model_tag, "validation_best")
            self.assertIsNotNone(load_conf.loaded_combo_scorer_path)

    def test_balanced_best_checkpoint_loads_combo_scorer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            conf = Config()
            conf.model_dir = tmpdir
            conf.load_model = False
            agents = Agents(conf)
            summary = {
                "final_pulse": 600.0,
                "smoothness_index": 5.0,
                "finished_order_count": end2end.pro_num,
            }
            save_tagged_model(agents, conf, summary, 625.0, "balanced_best")

            load_conf = Config()
            load_conf.model_dir = tmpdir
            load_conf.load_model = True
            load_conf.model_tag = "balanced_best"
            Agents(load_conf)
            self.assertEqual(load_conf.loaded_model_tag, "balanced_best")
            self.assertIsNotNone(load_conf.loaded_combo_scorer_path)

    def test_end2end_load_model_env_supports_resume(self):
        old_load_model = os.environ.get("END2END_LOAD_MODEL")
        old_model_tag = os.environ.get("END2END_MODEL_TAG")
        try:
            os.environ["END2END_LOAD_MODEL"] = "1"
            os.environ["END2END_MODEL_TAG"] = "validation_best"
            conf = Config()
            self.assertTrue(conf.load_model)
            self.assertEqual(conf.model_tag, "validation_best")
        finally:
            if old_load_model is None:
                os.environ.pop("END2END_LOAD_MODEL", None)
            else:
                os.environ["END2END_LOAD_MODEL"] = old_load_model
            if old_model_tag is None:
                os.environ.pop("END2END_MODEL_TAG", None)
            else:
                os.environ["END2END_MODEL_TAG"] = old_model_tag


if __name__ == "__main__":
    unittest.main()
