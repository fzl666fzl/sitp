import contextlib
import io
import os

from GNN_QATTEN_diagnostics import (
    DEFAULT_AUX010_MODEL_DIR,
    changed_decision_details,
    compare_station_actions,
    run_episode,
    run_qatten_baseline,
    summarize_records,
)


def _quiet_call(func, *args, **kwargs):
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        return func(*args, **kwargs)


def _actions(summary):
    return [
        (item.get("station_id"), item.get("proc_rule"), item.get("team_rule"))
        for item in summary.get("station_actions", [])
    ]


def _format_actions(summary):
    return ";".join(
        f"s{station}:a0={proc},a1={team}"
        for station, proc, team in _actions(summary)
    )


def _result_row(label, summary, diagnostics=None, executed_compare=None):
    diagnostics = diagnostics or {}
    executed_compare = executed_compare or {}
    return {
        "label": label,
        "final_pulse": summary.get("final_pulse"),
        "SI": summary.get("smoothness_index"),
        "actions": _format_actions(summary),
        "agent0_executed_action_change_rate_vs_zero": executed_compare.get(
            "agent0_executed_action_change_rate"
        ),
        "changed_decision_count": executed_compare.get(
            "agent0_executed_changed_count", diagnostics.get("argmax_changed_count")
        ),
        "rerank_candidate_size_mean": diagnostics.get("rerank_candidate_size_mean"),
        "rerank_candidate_size_per_decision": diagnostics.get(
            "rerank_candidate_size_per_decision"
        ),
        "changed_only_when_margin_below_threshold": diagnostics.get(
            "changed_only_when_margin_below_threshold"
        ),
        "changed_action_q_gap_mean": diagnostics.get("changed_action_q_gap_mean"),
        "changed_action_bias_gain_mean": diagnostics.get("changed_action_bias_gain_mean"),
        "changed_action_target_gain_mean": diagnostics.get("changed_action_target_gain_mean"),
        "changed_action_load_penalty_gain_mean": diagnostics.get(
            "changed_action_load_penalty_gain_mean"
        ),
        "selected_successor_time_higher_rate": diagnostics.get(
            "selected_successor_time_higher_than_original_rate"
        ),
    }


def _print_row(row):
    fields = [
        "label",
        "final_pulse",
        "SI",
        "actions",
        "agent0_executed_action_change_rate_vs_zero",
        "changed_decision_count",
        "rerank_candidate_size_mean",
        "rerank_candidate_size_per_decision",
        "changed_only_when_margin_below_threshold",
        "changed_action_q_gap_mean",
        "changed_action_bias_gain_mean",
        "changed_action_target_gain_mean",
        "changed_action_load_penalty_gain_mean",
        "selected_successor_time_higher_rate",
    ]
    print(",".join(str(row.get(field)) for field in fields))


def _run_normal(label, zero_summary, **kwargs):
    normal_conf, normal_summary, records = _quiet_call(
        run_episode, zero_gnn=False, collect_diagnostics=True, **kwargs
    )
    diagnostics = summarize_records(records)
    executed_compare = compare_station_actions(normal_summary, zero_summary)
    row = _result_row(label, normal_summary, diagnostics, executed_compare)
    details = changed_decision_details(records, limit=3)
    return normal_conf, row, details


def main():
    model_dir = os.path.abspath(os.getenv("ACTION_GNN_MODEL_DIR", DEFAULT_AUX010_MODEL_DIR))
    model_tag = os.getenv("ACTION_GNN_MODEL_TAG", "latest")

    qatten_conf, qatten_summary = _quiet_call(run_qatten_baseline)
    zero_conf, zero_summary, _ = _quiet_call(
        run_episode,
        zero_gnn=True,
        collect_diagnostics=False,
        fusion_mode="margin_gated",
        margin_threshold=0.5,
        rerank_weight=0.5,
        model_dir=model_dir,
        model_tag=model_tag,
    )

    checkpoint_root = model_dir if os.path.basename(model_dir) == zero_conf.map_name else os.path.join(model_dir, zero_conf.map_name)
    print("checkpoint:", checkpoint_root)
    print("model_tag:", model_tag)
    print(
        "label,final_pulse,SI,actions,agent0_executed_action_change_rate_vs_zero,"
        "changed_decision_count,rerank_candidate_size_mean,rerank_candidate_size_per_decision,"
        "changed_only_when_margin_below_threshold,changed_action_q_gap_mean,"
        "changed_action_bias_gain_mean,changed_action_target_gain_mean,"
        "changed_action_load_penalty_gain_mean,selected_successor_time_higher_rate"
    )
    _print_row(_result_row("baseline_qatten", qatten_summary))
    _print_row(_result_row("aux010_zero_gnn", zero_summary))

    rows = []
    details_by_label = {}

    _, row, details = _run_normal(
        "aux010_add_bias_w0.5",
        zero_summary,
        bias_weight=0.5,
        fusion_mode="add_bias",
        model_dir=model_dir,
        model_tag=model_tag,
    )
    rows.append(row)
    details_by_label[row["label"]] = details
    _print_row(row)

    for threshold in [0.2, 0.5, 1.0]:
        for weight in [0.2, 0.5, 1.0]:
            label = f"aux010_margin_t{threshold}_w{weight}"
            _, row, details = _run_normal(
                label,
                zero_summary,
                fusion_mode="margin_gated",
                margin_threshold=threshold,
                rerank_weight=weight,
                model_dir=model_dir,
                model_tag=model_tag,
            )
            rows.append(row)
            details_by_label[label] = details
            _print_row(row)

    for topk in [2, 3]:
        for weight in [0.2, 0.5, 1.0]:
            label = f"aux010_topk{topk}_w{weight}"
            _, row, details = _run_normal(
                label,
                zero_summary,
                fusion_mode="topk_rerank",
                topk=topk,
                rerank_weight=weight,
                model_dir=model_dir,
                model_tag=model_tag,
            )
            rows.append(row)
            details_by_label[label] = details
            _print_row(row)

    print("\nchanged_decision_details:")
    for row in rows:
        details = details_by_label[row["label"]]
        if details:
            print(row["label"], details)

    ranked = sorted(
        rows,
        key=lambda item: (
            item["SI"] if item["SI"] is not None else float("inf"),
            item["final_pulse"] if item["final_pulse"] is not None else float("inf"),
        ),
    )
    if ranked:
        print("\nbest_by_SI:", ranked[0])
    print("baseline_qatten_loaded_checkpoint:", getattr(qatten_conf, "loaded_model_tag", None))
    print("aux010_loaded_checkpoint:", getattr(zero_conf, "loaded_model_tag", None))
    print("aux010_loaded_gnn_path:", getattr(zero_conf, "loaded_gnn_path", None))


if __name__ == "__main__":
    main()
