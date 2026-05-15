import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import Agents
from config_gnn_qatten_end2end import Config
from GNN_QATTEN_END2END_online import configure_online_defaults, online_init_pulse
import rollout_end2end


def set_eval_seed(conf):
    seed = int(os.getenv("END2END_ONLINE_SEED", str(getattr(conf, "end2end_validation_seed", conf.seed))))
    conf.seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def output_dir():
    path = os.getenv("END2END_GANTT_DIR", "figures_end2end")
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(__file__), path)
    os.makedirs(path, exist_ok=True)
    return path


def output_prefix(conf):
    tag = str(getattr(conf, "model_tag", "online_pulse_best"))
    seed = str(getattr(conf, "seed", "unknown"))
    return "gantt_{}_seed{}".format(tag, seed)


def run_online_summary():
    conf = configure_online_defaults(Config())
    set_eval_seed(conf)
    agents = Agents(conf)
    init_pulse = online_init_pulse(conf)
    _, _, _, summary = rollout_end2end.generate_episode(
        agents,
        conf,
        [],
        init_pulse,
        0,
        [],
        evaluate=True,
    )
    return conf, init_pulse, summary


def station_team_label(station_id, team_id):
    return "S{}-T{}".format(int(station_id) + 1, int(team_id) + 1)


def plot_gantt(conf, init_pulse, summary, png_path):
    actions = list(summary.get("actions") or [])
    if not actions:
        raise ValueError("summary has no actions to plot")

    lanes = []
    for station_id in range(rollout_end2end.station_num):
        for team_id in range(rollout_end2end.team_num):
            lanes.append((station_id, team_id))
    lane_to_y = {lane: idx for idx, lane in enumerate(lanes)}

    colors = plt.get_cmap("tab20").colors
    fig_height = max(6.0, len(lanes) * 0.48)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    for idx, item in enumerate(actions):
        station_id = int(item["station_id"])
        team_id = int(item["team_id"])
        start = float(item["start_time"])
        finish = float(item["finish_time"])
        duration = max(finish - start, 0.0)
        y = lane_to_y[(station_id, team_id)]
        color = colors[int(item["proc_id"]) % len(colors)]
        ax.barh(
            y,
            duration,
            left=start,
            height=0.72,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            alpha=0.88,
        )
        label = "P{}".format(int(item["proc_id"]))
        ax.text(
            start + duration / 2.0,
            y,
            label,
            ha="center",
            va="center",
            fontsize=7,
            color="black",
            clip_on=True,
        )

    max_finish = max(float(item["finish_time"]) for item in actions)
    for station_id in range(rollout_end2end.station_num + 1):
        boundary = station_id * float(init_pulse)
        ax.axvline(boundary, color="#333333", linestyle="--", linewidth=0.8, alpha=0.6)
        if station_id < rollout_end2end.station_num:
            ax.text(
                boundary + 4,
                len(lanes) - 0.35,
                "S{} start".format(station_id + 1),
                rotation=90,
                va="top",
                fontsize=8,
                color="#333333",
            )

    y_labels = [station_team_label(station_id, team_id) for station_id, team_id in lanes]
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Time")
    ax.set_ylabel("Station-Team")
    ax.set_xlim(0, max(max_finish, rollout_end2end.station_num * float(init_pulse)) * 1.03)
    ax.grid(axis="x", color="#d0d0d0", linestyle="-", linewidth=0.5, alpha=0.6)
    title = (
        "End-to-end GNN-QATTEN Gantt | tag={} | pulse={:.1f} | SI={:.4f}"
        .format(
            getattr(conf, "loaded_model_tag", getattr(conf, "model_tag", "")),
            float(summary.get("final_pulse", 0.0)),
            float(summary.get("smoothness_index", 0.0)),
        )
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def main():
    conf, init_pulse, summary = run_online_summary()
    out_dir = output_dir()
    prefix = output_prefix(conf)
    png_path = os.path.join(out_dir, prefix + ".png")
    json_path = os.path.join(out_dir, prefix + "_summary.json")
    plot_gantt(conf, init_pulse, summary, png_path)
    with open(json_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)
    print("saved gantt:", png_path)
    print("saved summary:", json_path)
    print("final_pulse:", summary.get("final_pulse"))
    print("SI:", summary.get("smoothness_index"))
    print("station_times:", summary.get("station_times"))
    print("finished_order_count:", summary.get("finished_order_count"))


if __name__ == "__main__":
    main()
