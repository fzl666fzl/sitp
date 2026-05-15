import json
import math
import os
import random
import sys

import numpy as np
import torch

from agent import Agents
from config_gnn_qatten_end2end import Config
from GNN_QATTEN_END2END_online import (
    configure_online_defaults,
    online_init_pulse,
    set_eval_seed,
)
import rollout_end2end


def _summary_path(agents, conf):
    tag = str(getattr(conf, "model_tag", "validation_best"))
    return os.path.join(agents.policy.model_dir, f"{tag}_summary.json")


def _close_enough(left, right, tol=1e-6):
    return math.isclose(float(left), float(right), rel_tol=tol, abs_tol=tol)


def _check_equal(actual, expected):
    errors = []
    if int(actual.get("finished_order_count", 0)) != int(expected.get("finished_order_count", 0)):
        errors.append(
            "finished_order_count actual={} expected={}".format(
                actual.get("finished_order_count"), expected.get("finished_order_count")
            )
        )
    if not _close_enough(actual.get("final_pulse", 0.0), expected.get("final_pulse", 0.0)):
        errors.append(
            "final_pulse actual={} expected={}".format(
                actual.get("final_pulse"), expected.get("final_pulse")
            )
        )
    if not _close_enough(
        actual.get("smoothness_index", 0.0), expected.get("smoothness_index", 0.0)
    ):
        errors.append(
            "smoothness_index actual={} expected={}".format(
                actual.get("smoothness_index"), expected.get("smoothness_index")
            )
        )
    actual_times = actual.get("station_times") or []
    expected_times = expected.get("station_times") or []
    if len(actual_times) != len(expected_times):
        errors.append(
            "station_times length actual={} expected={}".format(
                len(actual_times), len(expected_times)
            )
        )
    else:
        for idx, (actual_time, expected_time) in enumerate(zip(actual_times, expected_times)):
            if not _close_enough(actual_time, expected_time):
                errors.append(
                    "station_times[{}] actual={} expected={}".format(
                        idx, actual_time, expected_time
                    )
                )
    return errors


def main():
    sys.argv = sys.argv[:1]
    conf = configure_online_defaults(Config())
    set_eval_seed(conf)
    agents = Agents(conf)
    init_pulse = online_init_pulse(conf)

    summary_path = _summary_path(agents, conf)
    if not os.path.exists(summary_path):
        raise FileNotFoundError("summary file not found: {}".format(summary_path))
    with open(summary_path, "r", encoding="utf-8") as file_obj:
        expected = json.load(file_obj)

    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    _, _, _, actual = rollout_end2end.generate_episode(
        agents,
        conf,
        [],
        init_pulse,
        0,
        [],
        evaluate=True,
    )

    errors = _check_equal(actual, expected)
    print("configured checkpoint:", conf.model_tag)
    print("loaded checkpoint:", getattr(conf, "loaded_model_tag", None))
    print("summary path:", summary_path)
    print("actual final_pulse:", actual.get("final_pulse"))
    print("actual SI:", actual.get("smoothness_index"))
    print("expected final_pulse:", expected.get("final_pulse"))
    print("expected SI:", expected.get("smoothness_index"))
    if errors:
        print("online consistency check failed:")
        for item in errors:
            print("  -", item)
        raise SystemExit(1)
    print("online consistency check passed")


if __name__ == "__main__":
    main()
