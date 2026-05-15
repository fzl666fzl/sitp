import os
import random

import numpy as np
import torch

from agent import Agents
from config import Config
import onlinerollout


def parse_seeds(value):
    value = (value or "100-149").strip()
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def evaluate_seed(agents, conf, seed, init_pulse):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    _, _, _, summary = onlinerollout.generate_episode(
        agents, conf, [], init_pulse, 0, [], evaluate=True
    )
    return summary


def main():
    seeds = parse_seeds(os.getenv("QATTEN_EVAL_SEEDS", "100-149"))
    init_pulse = int(os.getenv("QATTEN_EVAL_INIT_PULSE", "608"))
    model_tag = os.getenv("QATTEN_EVAL_MODEL_TAG", "5")

    conf = Config()
    conf.load_model = True
    conf.model_tag = model_tag
    agents = Agents(conf)

    results = []
    for seed in seeds:
        summary = evaluate_seed(agents, conf, seed, init_pulse)
        results.append(summary)
        print(
            "seed={seed} final_pulse={pulse} SI={si} station_times={station_times} "
            "finished={finished}".format(
                seed=seed,
                pulse=summary.get("final_pulse"),
                si=summary.get("smoothness_index"),
                station_times=summary.get("station_times"),
                finished=summary.get("finished_order_count", "n/a"),
            )
        )

    pulses = np.array([item.get("final_pulse", 0.0) for item in results], dtype=float)
    si_values = np.array([item.get("smoothness_index", 0.0) for item in results], dtype=float)
    print("tag =", model_tag)
    print("loaded_tag =", getattr(conf, "loaded_model_tag", None))
    print("n =", len(results))
    print("pulse mean =", float(np.mean(pulses)))
    print("pulse median =", float(np.median(pulses)))
    print("pulse min =", float(np.min(pulses)))
    print("pulse max =", float(np.max(pulses)))
    print("SI mean =", float(np.mean(si_values)))
    print("SI median =", float(np.median(si_values)))
    print("SI min =", float(np.min(si_values)))
    print("SI max =", float(np.max(si_values)))
    print("SI <= 5 =", int(np.sum(si_values <= 5)))
    print("SI > 10 =", int(np.sum(si_values > 10)))


if __name__ == "__main__":
    main()
