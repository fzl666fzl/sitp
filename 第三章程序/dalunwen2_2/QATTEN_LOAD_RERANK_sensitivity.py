import contextlib
import io

from QATTEN_LOAD_RERANK_online import format_actions, run_online, summarize_records


def quiet_run(**kwargs):
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        return run_online(**kwargs)


def print_row(label, conf, summary, records):
    diagnostics = summarize_records(records)
    print(
        ",".join(
            str(item)
            for item in [
                label,
                conf.qatten_load_rerank_mode,
                conf.qatten_load_margin_threshold,
                conf.qatten_load_topk,
                conf.qatten_load_penalty_weight,
                summary.get("final_pulse"),
                summary.get("smoothness_index"),
                format_actions(summary),
                diagnostics["changed_count"],
                diagnostics["changed_rate"],
                diagnostics["candidate_sizes"],
                diagnostics["changed_details"],
            ]
        )
    )


def main():
    print(
        "label,mode,margin_threshold,topk,penalty_weight,final_pulse,SI,"
        "actions,changed_count,changed_rate,candidate_sizes,changed_details"
    )

    conf, summary, records, _ = quiet_run(
        mode="margin_gated", threshold=1.0, penalty_weight=0.0
    )
    print_row("parity_lambda0", conf, summary, records)

    for threshold in [0.2, 0.5, 1.0]:
        for weight in [0.0, 0.05, 0.1, 0.2, 0.5]:
            label = "margin_t{}_lambda{}".format(threshold, weight)
            conf, summary, records, _ = quiet_run(
                mode="margin_gated",
                threshold=threshold,
                penalty_weight=weight,
            )
            print_row(label, conf, summary, records)

    for topk in [2, 3]:
        for weight in [0.0, 0.05, 0.1, 0.2, 0.5]:
            label = "topk{}_lambda{}".format(topk, weight)
            conf, summary, records, _ = quiet_run(
                mode="topk_rerank",
                topk=topk,
                penalty_weight=weight,
            )
            print_row(label, conf, summary, records)


if __name__ == "__main__":
    main()
