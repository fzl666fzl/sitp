import os
import random

import numpy as np
import torch

from agent import Agents
from config_gnn_qatten_end2end import Config
import rollout_end2end


def eval_init_pulse():
    return int(os.getenv("END2END_EVAL_INIT_PULSE", "608"))


def eval_tags():
    raw_tags = os.getenv(
        "END2END_EVAL_TAGS",
        "expert_trace_best,online_balanced_best,online_pulse_best,balanced_best,validation_best,best,latest",
    )
    return tuple(tag.strip() for tag in raw_tags.split(",") if tag.strip())


def set_eval_seed(conf):
    conf.seed = int(os.getenv("END2END_EVAL_SEED", str(conf.seed)))
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)


def evaluate_tag(tag):
    conf = Config()
    conf.load_model = True
    conf.model_tag = tag
    set_eval_seed(conf)
    agents = Agents(conf)
    _, _, _, summary = rollout_end2end.generate_episode(
        agents, conf, [], eval_init_pulse(), 0, [], evaluate=True
    )
    final_pulse = summary.get("final_pulse")
    si = summary.get("smoothness_index")
    score = None
    if final_pulse is not None and si is not None:
        score = float(final_pulse) + float(si)
    return {
        "tag": tag,
        "loaded_tag": getattr(conf, "loaded_model_tag", None),
        "final_pulse": final_pulse,
        "SI": si,
        "score": score,
        "station_times": summary.get("station_times"),
        "finished_order_count": summary.get("finished_order_count"),
    }


def main():
    for tag in eval_tags():
        result = evaluate_tag(tag)
        print(
            "tag={tag} loaded={loaded_tag} final_pulse={final_pulse} SI={SI} score={score} "
            "station_times={station_times} finished={finished_order_count}".format(**result)
        )


if __name__ == "__main__":
    main()
