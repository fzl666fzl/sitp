import random

import numpy as np
import torch

from agent import Agents
from config_gnn_qatten import Config
import onlinerollout
from utils import ReplayBuffer


INIT_PULSE = 608


def set_eval_seed(conf):
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)


def run_episode(zero_gnn=False, collect_diagnostics=False, bias_weight=None):
    conf = Config()
    conf.load_model = True
    conf.zero_gnn_embedding = zero_gnn
    if bias_weight is not None:
        conf.gnn_action_bias_weight = float(bias_weight)
    conf.record_gnn_diagnostics = collect_diagnostics
    conf.gnn_diagnostic_records = [] if collect_diagnostics else None
    set_eval_seed(conf)

    agents = Agents(conf)
    buffer = ReplayBuffer(conf)
    episode, _, _, summary = onlinerollout.generate_episode(
        agents, conf, [], INIT_PULSE, 0, [], evaluate=True
    )
    buffer.store_episode(episode)
    return conf, summary, conf.gnn_diagnostic_records or []


def summarize_records(records):
    if not records:
        return {
            "total_decisions": 0,
            "q_changed_count": 0,
            "argmax_changed_count": 0,
            "executed_action_changed_count": 0,
            "mean_abs_q_delta": None,
            "action_change_rate": None,
            "agent0_action_change_rate": None,
            "agent0_executed_action_change_rate": None,
            "mean_margin": None,
            "mean_q_delta_to_margin": None,
        }

    agent0_records = [item for item in records if item.get("agent_id") == 0]
    deltas = [item.get("mean_abs_q_delta", 0.0) for item in records]
    changes = [item.get("action_changed", 0) for item in records]
    agent0_changes = [item.get("action_changed", 0) for item in agent0_records]
    agent0_executed_changes = [
        item.get("executed_action_changed", 0) for item in agent0_records
    ]
    return {
        "total_decisions": len(agent0_records),
        "q_changed_count": int(sum(item.get("q_changed", 0) for item in agent0_records)),
        "argmax_changed_count": int(sum(agent0_changes)),
        "executed_action_changed_count": int(sum(agent0_executed_changes)),
        "mean_abs_q_delta": round(float(np.mean(deltas)), 6),
        "action_change_rate": round(float(np.mean(changes)), 6),
        "agent0_action_change_rate": round(float(np.mean(agent0_changes)), 6)
        if agent0_changes
        else None,
        "agent0_executed_action_change_rate": round(float(np.mean(agent0_executed_changes)), 6)
        if agent0_executed_changes
        else None,
        "mean_margin": round(
            float(np.mean([item.get("q_margin_zero", 0.0) for item in agent0_records])), 6
        )
        if agent0_records
        else None,
        "mean_q_delta_to_margin": round(
            float(np.mean([item.get("q_delta_to_margin", 0.0) for item in agent0_records])), 6
        )
        if agent0_records
        else None,
    }


def compare_station_actions(normal_summary, zero_summary):
    normal_actions = normal_summary.get("station_actions") or []
    zero_actions = zero_summary.get("station_actions") or []
    paired_count = min(len(normal_actions), len(zero_actions))
    if paired_count == 0:
        return {
            "paired_station_count": 0,
            "agent0_executed_changed_count": 0,
            "executed_action_changed_count": 0,
            "agent0_executed_action_change_rate": None,
            "executed_action_change_rate": None,
        }

    agent0_changed = 0
    any_agent_changed = 0
    for normal_item, zero_item in zip(normal_actions, zero_actions):
        proc_changed = normal_item.get("proc_rule") != zero_item.get("proc_rule")
        team_changed = normal_item.get("team_rule") != zero_item.get("team_rule")
        agent0_changed += int(proc_changed)
        any_agent_changed += int(proc_changed or team_changed)
    return {
        "paired_station_count": paired_count,
        "agent0_executed_changed_count": agent0_changed,
        "executed_action_changed_count": any_agent_changed,
        "agent0_executed_action_change_rate": round(agent0_changed / paired_count, 6),
        "executed_action_change_rate": round(any_agent_changed / paired_count, 6),
    }


def run_online_pair(bias_weight=None):
    normal_conf, normal_summary, records = run_episode(
        zero_gnn=False, collect_diagnostics=True, bias_weight=bias_weight
    )
    zero_conf, zero_summary, _ = run_episode(
        zero_gnn=True, collect_diagnostics=False, bias_weight=bias_weight
    )
    diagnostics = summarize_records(records)
    executed_compare = compare_station_actions(normal_summary, zero_summary)
    return {
        "bias_weight": float(normal_conf.gnn_action_bias_weight),
        "normal_conf": normal_conf,
        "zero_conf": zero_conf,
        "normal_summary": normal_summary,
        "zero_summary": zero_summary,
        "records": records,
        "diagnostics": diagnostics,
        "executed_compare": executed_compare,
    }


def print_changed_decisions(records, limit=3):
    changed = [
        item for item in records
        if item.get("agent_id") == 0 and item.get("action_changed")
    ][:limit]
    if not changed:
        print("agent0 changed decision samples: none")
        return
    print("agent0 changed decision samples:")
    for item in changed:
        print({
            "step": item.get("step"),
            "station": item.get("station_id"),
            "episode_time": item.get("episode_time"),
            "q_zero": item.get("q_zero"),
            "gnn_bias": item.get("gnn_bias"),
            "q_normal": item.get("q_normal"),
            "zero_action": item.get("zero_action"),
            "normal_action": item.get("normal_action"),
            "executed_action": item.get("executed_action"),
            "q_margin_zero": round(float(item.get("q_margin_zero", 0.0)), 6),
            "max_abs_bias": round(float(item.get("max_abs_bias", 0.0)), 6),
            "q_delta_to_margin": round(float(item.get("q_delta_to_margin", 0.0)), 6),
            "rule_node_ids": item.get("rule_node_ids"),
            "selected_rule_node_id": item.get("selected_rule_node_id"),
            "valid_order_count": item.get("valid_order_count"),
            "planned_order_ids": item.get("planned_order_ids"),
            "finished_order_ids": item.get("finished_order_ids"),
        })


def main():
    result = run_online_pair()
    normal_conf = result["normal_conf"]
    zero_conf = result["zero_conf"]
    normal_summary = result["normal_summary"]
    zero_summary = result["zero_summary"]
    diagnostics = result["diagnostics"]
    executed_compare = result["executed_compare"]

    print("\n===== GNN Qatten online-paired diagnostics =====")
    print("diagnostic path: online rollout counterfactual Q + separate zero online rollout")
    print("bias_weight:", result["bias_weight"])
    print("normal loaded checkpoint:", getattr(normal_conf, "loaded_model_tag", None))
    print("zero loaded checkpoint:", getattr(zero_conf, "loaded_model_tag", None))
    print("normal final_pulse:", normal_summary.get("final_pulse"))
    print("normal smoothness_index:", normal_summary.get("smoothness_index"))
    print("zero final_pulse:", zero_summary.get("final_pulse"))
    print("zero smoothness_index:", zero_summary.get("smoothness_index"))
    print("mean_abs_q_delta:", diagnostics["mean_abs_q_delta"])
    print("action_change_rate:", diagnostics["action_change_rate"])
    print("agent0_action_change_rate:", diagnostics["agent0_action_change_rate"])
    print("agent0_executed_action_change_rate:", diagnostics["agent0_executed_action_change_rate"])
    print("mean_margin:", diagnostics["mean_margin"])
    print("mean_q_delta_to_margin:", diagnostics["mean_q_delta_to_margin"])
    print("trace_counts:", {
        "total_decisions": diagnostics["total_decisions"],
        "q_changed_count": diagnostics["q_changed_count"],
        "argmax_changed_count": diagnostics["argmax_changed_count"],
        "executed_action_changed_count": diagnostics["executed_action_changed_count"],
        "agent0_changed_count": diagnostics["argmax_changed_count"],
    })
    print("paired_actual_execution:", executed_compare)
    print("normal station_actions:", normal_summary.get("station_actions"))
    print("zero station_actions:", zero_summary.get("station_actions"))
    print_changed_decisions(result["records"])


if __name__ == "__main__":
    main()
