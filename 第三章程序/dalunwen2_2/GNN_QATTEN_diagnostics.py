import random
import os

import numpy as np
import torch

from agent import Agents
from config import Config as QattenConfig
from config_gnn_qatten import Config
import onlinerollout
from utils import ReplayBuffer


INIT_PULSE = 608
DEFAULT_AUX010_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "models_gnn_qatten_action_bias_aux010"
)


def set_eval_seed(conf):
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)


def _configure_aux010_gnn(conf, model_dir=None, model_tag=None):
    conf.use_gnn_graph_embedding = False
    conf.use_gnn_action_bias = True
    conf.use_gnn = True
    conf.gnn_aux_weight = 0.0
    conf.model_dir = os.path.abspath(
        model_dir or os.getenv("ACTION_GNN_MODEL_DIR", DEFAULT_AUX010_MODEL_DIR)
    )
    conf.model_tag = model_tag or os.getenv("ACTION_GNN_MODEL_TAG", "latest")


def run_episode(
    zero_gnn=False,
    collect_diagnostics=False,
    bias_weight=None,
    fusion_mode=None,
    margin_threshold=None,
    topk=None,
    rerank_weight=None,
    use_load_penalty=None,
    load_penalty_weight=None,
    model_dir=None,
    model_tag=None,
):
    conf = Config()
    conf.load_model = True
    _configure_aux010_gnn(conf, model_dir=model_dir, model_tag=model_tag)
    conf.zero_gnn_embedding = zero_gnn
    if bias_weight is not None:
        conf.gnn_action_bias_weight = float(bias_weight)
    if fusion_mode is not None:
        conf.gnn_action_fusion_mode = fusion_mode
    if margin_threshold is not None:
        conf.gnn_margin_threshold = float(margin_threshold)
    if topk is not None:
        conf.gnn_topk = int(topk)
    if rerank_weight is not None:
        conf.gnn_rerank_weight = float(rerank_weight)
    if use_load_penalty is not None:
        conf.gnn_use_load_penalty = bool(use_load_penalty)
    if load_penalty_weight is not None:
        conf.gnn_load_penalty_weight = float(load_penalty_weight)
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


def run_qatten_baseline():
    conf = QattenConfig()
    conf.load_model = True
    set_eval_seed(conf)

    agents = Agents(conf)
    buffer = ReplayBuffer(conf)
    episode, _, _, summary = onlinerollout.generate_episode(
        agents, conf, [], INIT_PULSE, 0, [], evaluate=True
    )
    buffer.store_episode(episode)
    return conf, summary


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
            "candidate_bias_std": None,
            "candidate_target_std": None,
            "candidate_target_bias_corr": None,
            "candidate_pairwise_acc": None,
            "mean_agent0_q_margin": None,
            "mean_scaled_bias_abs": None,
            "mean_scaled_bias_to_margin": None,
            "rerank_candidate_size_mean": None,
            "rerank_candidate_size_per_decision": [],
            "changed_only_when_margin_below_threshold": None,
            "changed_action_q_gap_mean": None,
            "changed_action_bias_gain_mean": None,
            "changed_action_target_gain_mean": None,
            "changed_action_load_penalty_gain_mean": None,
            "changed_action_pred_si_gain_mean": None,
            "selected_successor_time_higher_than_original_rate": None,
            "selected_pred_si_lower_than_original_rate": None,
        }

    agent0_records = [item for item in records if item.get("agent_id") == 0]
    def mean_metric(key):
        values = [item.get(key) for item in agent0_records if item.get(key) is not None]
        return round(float(np.mean(values)), 6) if values else None

    changed_records = [item for item in agent0_records if item.get("action_changed")]
    margin_records = [item for item in agent0_records if item.get("fusion_mode") == "margin_gated"]
    changed_only_when_margin_below_threshold = None
    if margin_records:
        changed_only_when_margin_below_threshold = all(
            item.get("changed_action_q_gap") is not None
            and item.get("changed_action_q_gap") <= float(item.get("rerank_margin_threshold", 0.0)) + 1e-8
            for item in changed_records
        )
    target_gain_flags = [
        item.get("selected_successor_time_higher_than_original")
        for item in changed_records
        if item.get("selected_successor_time_higher_than_original") is not None
    ]
    pred_si_gain_flags = [
        item.get("selected_pred_si_lower_than_original")
        for item in changed_records
        if item.get("selected_pred_si_lower_than_original") is not None
    ]
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
        "candidate_bias_std": mean_metric("candidate_bias_std"),
        "candidate_target_std": mean_metric("candidate_target_std"),
        "candidate_target_bias_corr": mean_metric("candidate_target_bias_corr"),
        "candidate_pairwise_acc": mean_metric("candidate_pairwise_acc"),
        "mean_agent0_q_margin": mean_metric("q_margin_zero"),
        "mean_scaled_bias_abs": mean_metric("max_abs_bias"),
        "mean_scaled_bias_to_margin": mean_metric("q_delta_to_margin"),
        "rerank_candidate_size_mean": mean_metric("rerank_candidate_size"),
        "rerank_candidate_size_per_decision": [
            item.get("rerank_candidate_size") for item in agent0_records
        ],
        "changed_only_when_margin_below_threshold": changed_only_when_margin_below_threshold,
        "changed_action_q_gap_mean": mean_metric("changed_action_q_gap"),
        "changed_action_bias_gain_mean": mean_metric("changed_action_bias_gain"),
        "changed_action_target_gain_mean": mean_metric("changed_action_target_gain"),
        "changed_action_load_penalty_gain_mean": mean_metric(
            "changed_action_load_penalty_gain"
        ),
        "changed_action_pred_si_gain_mean": mean_metric(
            "changed_action_pred_si_gain"
        ),
        "selected_successor_time_higher_than_original_rate": round(
            float(np.mean(target_gain_flags)), 6
        )
        if target_gain_flags
        else None,
        "selected_pred_si_lower_than_original_rate": round(
            float(np.mean(pred_si_gain_flags)), 6
        )
        if pred_si_gain_flags
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


def run_online_pair(
    bias_weight=None,
    fusion_mode=None,
    margin_threshold=None,
    topk=None,
    rerank_weight=None,
    use_load_penalty=None,
    load_penalty_weight=None,
    model_dir=None,
    model_tag=None,
):
    normal_conf, normal_summary, records = run_episode(
        zero_gnn=False,
        collect_diagnostics=True,
        bias_weight=bias_weight,
        fusion_mode=fusion_mode,
        margin_threshold=margin_threshold,
        topk=topk,
        rerank_weight=rerank_weight,
        use_load_penalty=use_load_penalty,
        load_penalty_weight=load_penalty_weight,
        model_dir=model_dir,
        model_tag=model_tag,
    )
    zero_conf, zero_summary, _ = run_episode(
        zero_gnn=True,
        collect_diagnostics=False,
        bias_weight=bias_weight,
        fusion_mode=fusion_mode,
        margin_threshold=margin_threshold,
        topk=topk,
        rerank_weight=rerank_weight,
        use_load_penalty=use_load_penalty,
        load_penalty_weight=load_penalty_weight,
        model_dir=model_dir,
        model_tag=model_tag,
    )
    diagnostics = summarize_records(records)
    executed_compare = compare_station_actions(normal_summary, zero_summary)
    return {
        "bias_weight": float(normal_conf.gnn_action_bias_weight),
        "fusion_mode": normal_conf.gnn_action_fusion_mode,
        "margin_threshold": float(getattr(normal_conf, "gnn_margin_threshold", 0.5)),
        "topk": int(getattr(normal_conf, "gnn_topk", 2)),
        "rerank_weight": float(getattr(normal_conf, "gnn_rerank_weight", 0.5)),
        "use_load_penalty": bool(getattr(normal_conf, "gnn_use_load_penalty", False)),
        "load_penalty_weight": float(getattr(normal_conf, "gnn_load_penalty_weight", 0.0)),
        "normal_conf": normal_conf,
        "zero_conf": zero_conf,
        "normal_summary": normal_summary,
        "zero_summary": zero_summary,
        "records": records,
        "diagnostics": diagnostics,
        "executed_compare": executed_compare,
    }


def changed_decision_details(records, limit=3):
    changed = [
        item for item in records
        if item.get("agent_id") == 0 and item.get("action_changed")
    ][:limit]
    details = []
    for item in changed:
        details.append({
            "step": item.get("step"),
            "station": item.get("station_id"),
            "q_margin": round(float(item.get("q_margin_zero", 0.0)), 6),
            "original_action": item.get("original_action", item.get("zero_action")),
            "rerank_action": item.get("rerank_action", item.get("normal_action")),
            "bias_values": item.get("gnn_bias"),
            "candidate_size": item.get("rerank_candidate_size"),
            "candidate_actions": item.get("rerank_candidate_actions"),
            "q_gap": item.get("changed_action_q_gap"),
            "bias_gain": item.get("changed_action_bias_gain"),
            "target_gain": item.get("changed_action_target_gain"),
            "load_penalty_gain": item.get("changed_action_load_penalty_gain"),
            "pred_si_gain": item.get("changed_action_pred_si_gain"),
            "selected_pred_si_lower_than_original": item.get(
                "selected_pred_si_lower_than_original"
            ),
            "predicted_si_by_action": item.get("predicted_si_by_action"),
            "load_penalty_values": item.get("load_penalty_values"),
            "successor_time_higher": item.get("selected_successor_time_higher_than_original"),
        })
    return details


def print_changed_decisions(records, limit=3):
    details = changed_decision_details(records, limit=limit)
    if not details:
        print("agent0 changed decision samples: none")
        return
    print("agent0 changed decision samples:")
    for item in details:
        print(item)


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
    print("fusion_mode:", result["fusion_mode"])
    print("bias_weight:", result["bias_weight"])
    print("rerank_weight:", result["rerank_weight"])
    print("use_load_penalty:", result["use_load_penalty"])
    print("load_penalty_weight:", result["load_penalty_weight"])
    print("margin_threshold:", result["margin_threshold"])
    print("topk:", result["topk"])
    print("normal loaded checkpoint:", getattr(normal_conf, "loaded_model_tag", None))
    print("normal loaded model_dir:", getattr(normal_conf, "model_dir", None))
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
    print("candidate_bias_std:", diagnostics["candidate_bias_std"])
    print("candidate_target_std:", diagnostics["candidate_target_std"])
    print("candidate_target_bias_corr:", diagnostics["candidate_target_bias_corr"])
    print("candidate_pairwise_acc:", diagnostics["candidate_pairwise_acc"])
    print("mean_agent0_q_margin:", diagnostics["mean_agent0_q_margin"])
    print("mean_scaled_bias_abs:", diagnostics["mean_scaled_bias_abs"])
    print("mean_scaled_bias_to_margin:", diagnostics["mean_scaled_bias_to_margin"])
    print("rerank_candidate_size_mean:", diagnostics["rerank_candidate_size_mean"])
    print("rerank_candidate_size_per_decision:", diagnostics["rerank_candidate_size_per_decision"])
    print(
        "changed_only_when_margin_below_threshold:",
        diagnostics["changed_only_when_margin_below_threshold"],
    )
    print("changed_action_q_gap_mean:", diagnostics["changed_action_q_gap_mean"])
    print("changed_action_bias_gain_mean:", diagnostics["changed_action_bias_gain_mean"])
    print("changed_action_target_gain_mean:", diagnostics["changed_action_target_gain_mean"])
    print(
        "changed_action_load_penalty_gain_mean:",
        diagnostics["changed_action_load_penalty_gain_mean"],
    )
    print(
        "selected_successor_time_higher_than_original_rate:",
        diagnostics["selected_successor_time_higher_than_original_rate"],
    )
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
