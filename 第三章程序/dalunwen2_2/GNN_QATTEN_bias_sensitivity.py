from GNN_QATTEN_diagnostics import run_online_pair


def main():
    weights = [0, 0.1, 0.2, 0.5, 1.0]
    print("weight,normal_final_pulse,normal_SI,zero_final_pulse,zero_SI,agent0_action_change_rate,executed_action_change_rate,mean_q_delta_to_margin")
    for weight in weights:
        result = run_online_pair(bias_weight=weight)
        diagnostics = result["diagnostics"]
        executed_compare = result["executed_compare"]
        normal_summary = result["normal_summary"]
        zero_summary = result["zero_summary"]
        print(
            f"{weight},"
            f"{normal_summary.get('final_pulse')},"
            f"{normal_summary.get('smoothness_index')},"
            f"{zero_summary.get('final_pulse')},"
            f"{zero_summary.get('smoothness_index')},"
            f"{diagnostics.get('agent0_action_change_rate')},"
            f"{executed_compare.get('agent0_executed_action_change_rate')},"
            f"{diagnostics.get('mean_q_delta_to_margin')}"
        )


if __name__ == "__main__":
    main()
