import contextlib
import io
import os

from agent import Agents
from config_gnn_qatten_si_predict_rerank import Config
import onlinerollout
from utils import ReplayBuffer
from GNN_QATTEN_SI_PREDICT_RERANK_online import INIT_PULSE, set_eval_seed


def _quiet_episode(conf):
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        set_eval_seed(conf)
        agents = Agents(conf)
        buffer = ReplayBuffer(conf)
        episode, _, _, summary = onlinerollout.generate_episode(
            agents, conf, [], INIT_PULSE, 0, [], evaluate=True
        )
        buffer.store_episode(episode)
    return conf, summary


def _actions(summary):
    return ";".join(
        "s{}:a0={},a1={}".format(
            item.get("station_id"), item.get("proc_rule"), item.get("team_rule")
        )
        for item in summary.get("station_actions", [])
    )


def _run_case(label, mode, si_weight, threshold=None, topk=None):
    conf = Config()
    conf.load_model = True
    conf.gnn_action_fusion_mode = mode
    conf.gnn_rerank_weight = float(os.getenv("ACTION_GNN_RERANK_WEIGHT", "1.0"))
    conf.si_predict_penalty_weight = float(si_weight)
    conf.record_gnn_diagnostics = True
    conf.gnn_diagnostic_records = []
    if threshold is not None:
        conf.gnn_margin_threshold = float(threshold)
    if topk is not None:
        conf.gnn_topk = int(topk)

    conf, summary = _quiet_episode(conf)
    trace = summary.get("gnn_trace_summary") or {}
    return {
        "label": label,
        "mode": conf.gnn_action_fusion_mode,
        "threshold": getattr(conf, "gnn_margin_threshold", None),
        "topk": getattr(conf, "gnn_topk", None),
        "gnn_weight": getattr(conf, "gnn_rerank_weight", None),
        "si_penalty_weight": conf.si_predict_penalty_weight,
        "final_pulse": summary.get("final_pulse"),
        "SI": summary.get("smoothness_index"),
        "station_times": summary.get("station_times"),
        "actions": _actions(summary),
        "agent0_executed_action_change_rate": (
            trace.get("executed_action_changed_count") / trace.get("total_decisions")
            if trace.get("total_decisions")
            else None
        ),
        "changed_decision_count": trace.get("executed_action_changed_count"),
        "changed_action_pred_si_gain_mean": trace.get("changed_action_pred_si_gain_mean"),
        "selected_pred_si_lower_than_original_rate": trace.get(
            "selected_pred_si_lower_than_original_rate"
        ),
        "candidate_sizes": trace.get("rerank_candidate_size_per_decision"),
    }


def _print_row(row):
    fields = [
        "label",
        "mode",
        "threshold",
        "topk",
        "gnn_weight",
        "si_penalty_weight",
        "final_pulse",
        "SI",
        "agent0_executed_action_change_rate",
        "changed_decision_count",
        "changed_action_pred_si_gain_mean",
        "selected_pred_si_lower_than_original_rate",
        "candidate_sizes",
        "actions",
    ]
    print(",".join(str(row.get(field)) for field in fields))


def main():
    cases = [
        ("margin_t1.0_si0.0", "margin_gated", 0.0, 1.0, None),
        ("margin_t1.0_si0.05", "margin_gated", 0.05, 1.0, None),
        ("margin_t1.0_si0.1", "margin_gated", 0.1, 1.0, None),
        ("margin_t1.0_si0.2", "margin_gated", 0.2, 1.0, None),
        ("margin_t1.0_si0.5", "margin_gated", 0.5, 1.0, None),
        ("topk2_si0.0", "topk_rerank", 0.0, None, 2),
        ("topk2_si0.05", "topk_rerank", 0.05, None, 2),
        ("topk2_si0.1", "topk_rerank", 0.1, None, 2),
        ("topk2_si0.2", "topk_rerank", 0.2, None, 2),
    ]

    print(
        "label,mode,threshold,topk,gnn_weight,si_penalty_weight,final_pulse,SI,"
        "agent0_executed_action_change_rate,changed_decision_count,"
        "changed_action_pred_si_gain_mean,selected_pred_si_lower_than_original_rate,"
        "candidate_sizes,actions"
    )
    rows = []
    for label, mode, si_weight, threshold, topk in cases:
        row = _run_case(label, mode, si_weight, threshold=threshold, topk=topk)
        rows.append(row)
        _print_row(row)

    ranked = sorted(
        rows,
        key=lambda item: (
            item["SI"] if item["SI"] is not None else float("inf"),
            item["final_pulse"] if item["final_pulse"] is not None else float("inf"),
        ),
    )
    if ranked:
        print("\nbest_by_SI:", ranked[0])


if __name__ == "__main__":
    main()
