import math

import numpy as np
import torch


def _as_int_list(value):
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value if item not in (0, None)]
    if value == 0:
        return []
    return [int(value)]


def _normalize_columns(features):
    max_values = np.max(np.abs(features), axis=0)
    max_values[max_values == 0] = 1.0
    return features / max_values


def build_procedure_graph(args):
    pro_ids = [int(item) for item in args.pro_id]
    id_to_idx = {pro_id: idx for idx, pro_id in enumerate(pro_ids)}
    node_count = len(pro_ids)
    adjacency = np.eye(node_count, dtype=np.float32)

    for pro_id, preorders in args.dict_preorder.items():
        dst_idx = id_to_idx.get(int(pro_id))
        if dst_idx is None:
            continue
        for pre_id in _as_int_list(preorders):
            src_idx = id_to_idx.get(pre_id)
            if src_idx is None:
                continue
            adjacency[src_idx, dst_idx] = 1.0
            adjacency[dst_idx, src_idx] = 1.0

    for pro_id, postorders in args.dict_postorder.items():
        src_idx = id_to_idx.get(int(pro_id))
        if src_idx is None:
            continue
        for post_id in _as_int_list(postorders):
            dst_idx = id_to_idx.get(post_id)
            if dst_idx is None:
                continue
            adjacency[src_idx, dst_idx] = 1.0
            adjacency[dst_idx, src_idx] = 1.0

    degrees = adjacency.sum(axis=1)
    degrees[degrees == 0] = 1.0
    degree_inv_sqrt = np.power(degrees, -0.5)
    adjacency = degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]

    features = []
    for pro_id in pro_ids:
        pre_count = len(_as_int_list(args.dict_preorder.get(pro_id, [])))
        post_count = len(_as_int_list(args.dict_postorder.get(pro_id, [])))
        features.append([
            float(args.dict_time.get(pro_id, 0.0)),
            float(args.dict_isfirstprocedure.get(pro_id, 0.0)),
            float(pre_count),
            float(post_count),
            float(args.dict_postnum.get(pro_id, 0.0)),
            float(args.dict_posttime.get(pro_id, 0.0)),
            float(args.dict_postsinktime.get(pro_id, 0.0)),
        ])

    features = _normalize_columns(np.array(features, dtype=np.float32))
    return torch.tensor(adjacency, dtype=torch.float32), torch.tensor(features, dtype=torch.float32)
