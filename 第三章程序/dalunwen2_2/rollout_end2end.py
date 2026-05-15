import math
import random

import numpy as np
import simpy

import rollout_dis as base


pro_num = base.pro_num
team_num = base.team_num
station_num = base.station_num
pro_id = base.pro_id
freeorders = base.freeorders
dict_time = base.dict_time
dict_preorder = base.dict_preorder
dict_postorder = base.dict_postorder
dict_postnum = base.dict_postnum
dict_posttime = base.dict_posttime
dict_postsinktime = base.dict_postsinktime


def encode_combo_action(proc_idx, team_id, team_count=team_num):
    if proc_idx < 0 or team_id < 0 or team_id >= team_count:
        raise ValueError("invalid combo action component")
    return int(proc_idx) * int(team_count) + int(team_id)


def decode_combo_action(action_id, proc_ids=pro_id, team_count=team_num):
    action_id = int(action_id)
    team_count = int(team_count)
    proc_idx = action_id // team_count
    team_id = action_id % team_count
    if proc_idx < 0 or proc_idx >= len(proc_ids):
        raise ValueError("invalid combo action id")
    return int(proc_idx), int(proc_ids[proc_idx]), int(team_id)


def extract_action_ids_from_summary(summary):
    action_trace = summary.get("actions") or []
    action_ids = []
    for item in action_trace:
        if "action_id" in item:
            action_ids.append(int(item["action_id"]))
            continue
        if "proc_id" not in item or "team_id" not in item:
            raise ValueError("summary action trace is missing action_id or proc/team ids")
        proc_id = int(item["proc_id"])
        team_id = int(item["team_id"])
        if proc_id not in pro_id:
            raise ValueError("summary action trace contains unknown proc_id {}".format(proc_id))
        proc_idx = list(pro_id).index(proc_id)
        action_ids.append(encode_combo_action(proc_idx, team_id))
    return action_ids


def extract_disturbances_from_summary(summary):
    action_trace = summary.get("actions") or []
    return [float(item.get("disturbance", 0.0)) for item in action_trace]


def infer_teacher_init_pulse(summary, default_pulse=None):
    action_trace = summary.get("actions") or []
    episode_times = sorted({float(item.get("episode_time", 0.0)) for item in action_trace})
    positive_times = [item for item in episode_times if item > 0]
    if positive_times:
        return int(round(min(positive_times)))
    if default_pulse is not None:
        return int(default_pulse)
    return int(round(float(summary.get("final_pulse", 608.0))))


def _order_list(value):
    if value in (None, 0):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _station_id_from_time(now_time, thispulse):
    if now_time >= 0 and now_time < thispulse:
        return 0
    if now_time >= thispulse and now_time < 2 * thispulse:
        return 1
    if now_time >= 2 * thispulse and now_time < 3 * thispulse:
        return 2
    return 3


def _duration(proc_id, team_id):
    time_w = (int(team_id) - 1) / 10 + 1
    return float(dict_time[int(proc_id)]) * time_w


def _disturbance_time(nowstation, proc_id, disable_disturbance=False):
    if disable_disturbance:
        return 0.0
    if nowstation.id >= station_num - 1:
        return 0.0
    tw = random.random()
    if nowstation.id == 0 and tw < 0.1:
        return float(dict_time[int(proc_id)]) * tw
    return 0.0


def _init_order_stfi(air):
    return {
        int(item): [float(air.startingtime[item]), float(air.finishtime[item])]
        for item in pro_id
    }


def _is_ready(proc_id, finished_orders):
    if proc_id in freeorders:
        return True
    return all(int(preorder) in finished_orders for preorder in _order_list(dict_preorder[proc_id]))


def get_available_combo_orders(air, planned_orders=None, finished_orders=None):
    planned_orders = planned_orders or set()
    finished_orders = set(air.order_finish) if finished_orders is None else set(finished_orders)
    candidates = []
    for proc_id in pro_id:
        proc_id = int(proc_id)
        if air.isfinish[proc_id] != 0 or proc_id in planned_orders:
            continue
        if _is_ready(proc_id, finished_orders):
            candidates.append(proc_id)
    return candidates


def _predicted_window(proc_id, team_id, planned_team_finish, order_stfi):
    predecessor_finish = 0.0
    for preorder in _order_list(dict_preorder[proc_id]):
        predecessor_finish = max(predecessor_finish, float(order_stfi[int(preorder)][1]))
    start_time = max(float(planned_team_finish[int(team_id)]), predecessor_finish)
    finish_time = start_time + _duration(proc_id, team_id)
    return start_time, finish_time


def build_end2end_avail_mask(
    air,
    nowstation,
    planned_orders,
    planned_team_finish,
    thispulse,
    finished_orders=None,
    order_stfi=None,
    blocked_teams=None,
):
    finished_orders = set(air.order_finish) if finished_orders is None else set(finished_orders)
    order_stfi = _init_order_stfi(air) if order_stfi is None else order_stfi
    blocked_teams = blocked_teams or set()
    mask = np.zeros(pro_num * team_num, dtype=np.float32)
    station_boundary = float(nowstation.pulse * (nowstation.id + 1))

    for proc_idx, proc_id in enumerate(pro_id):
        proc_id = int(proc_id)
        if air.isfinish[proc_id] != 0 or proc_id in planned_orders:
            continue
        if not _is_ready(proc_id, finished_orders):
            continue
        for team_id in range(team_num):
            if team_id in blocked_teams:
                continue
            _, finish_time = _predicted_window(proc_id, team_id, planned_team_finish, order_stfi)
            if nowstation.id < station_num - 1 and finish_time > station_boundary:
                continue
            mask[encode_combo_action(proc_idx, team_id)] = 1.0
    return mask


def build_combo_pair_features(
    air,
    nowstation,
    planned_orders,
    planned_team_finish,
    thispulse,
    finished_orders=None,
    order_stfi=None,
    blocked_teams=None,
    avail_mask=None,
):
    finished_orders = set(air.order_finish) if finished_orders is None else set(finished_orders)
    order_stfi = _init_order_stfi(air) if order_stfi is None else order_stfi
    if avail_mask is None:
        avail_mask = build_end2end_avail_mask(
            air,
            nowstation,
            planned_orders,
            planned_team_finish,
            thispulse,
            finished_orders=finished_orders,
            order_stfi=order_stfi,
            blocked_teams=blocked_teams,
        )
    pulse = max(float(thispulse), 1.0)
    station_boundary = float(nowstation.pulse * (nowstation.id + 1))
    remaining_ratio = (pro_num - len(finished_orders)) / max(float(pro_num), 1.0)
    completed_ratio = len(finished_orders) / max(float(pro_num), 1.0)
    features = np.zeros((pro_num * team_num, 14), dtype=np.float32)

    for proc_idx, proc_id in enumerate(pro_id):
        proc_id = int(proc_id)
        for team_id in range(team_num):
            action_id = encode_combo_action(proc_idx, team_id)
            start_time, finish_time = _predicted_window(
                proc_id, team_id, planned_team_finish, order_stfi
            )
            duration = _duration(proc_id, team_id)
            slack = max(0.0, station_boundary - finish_time) / pulse
            overload = max(0.0, finish_time - station_boundary) / pulse
            features[action_id] = np.array(
                [
                    float(avail_mask[action_id] > 0),
                    duration / pulse,
                    start_time / pulse,
                    finish_time / pulse,
                    float(planned_team_finish[team_id]) / pulse,
                    nowstation.id / max(float(station_num - 1), 1.0),
                    team_id / max(float(team_num - 1), 1.0),
                    slack,
                    overload,
                    float(dict_postnum[proc_id]) / 10.0,
                    float(dict_posttime[proc_id]) / 1500.0,
                    float(dict_postsinktime[proc_id]) / 6000.0,
                    remaining_ratio,
                    completed_ratio,
                ],
                dtype=np.float32,
            )
    return features


def plan_one_combo_action(
    nowstation,
    air,
    action_id,
    planned_orders,
    planned_team_finish,
    order_stfi,
    thispulse,
    finished_orders=None,
    blocked_teams=None,
    allow_disturbance=True,
    forced_disturbance=None,
):
    finished_orders = set(air.order_finish) if finished_orders is None else finished_orders
    blocked_teams = blocked_teams or set()
    proc_idx, proc_id, team_id = decode_combo_action(action_id)
    mask = build_end2end_avail_mask(
        air,
        nowstation,
        planned_orders,
        planned_team_finish,
        thispulse,
        finished_orders=finished_orders,
        order_stfi=order_stfi,
        blocked_teams=blocked_teams,
    )
    if mask[int(action_id)] <= 0:
        return {"valid": False, "action_id": int(action_id), "proc_id": proc_id, "team_id": team_id}

    start_time, finish_time = _predicted_window(proc_id, team_id, planned_team_finish, order_stfi)
    if forced_disturbance is None:
        disturbance = _disturbance_time(
            nowstation, proc_id, disable_disturbance=not allow_disturbance
        )
    else:
        disturbance = float(forced_disturbance)
    finish_time += disturbance
    station_boundary = float(nowstation.pulse * (nowstation.id + 1))
    if nowstation.id < station_num - 1 and finish_time > station_boundary:
        blocked_teams.add(team_id)
        return {
            "valid": False,
            "blocked_team": True,
            "action_id": int(action_id),
            "proc_id": int(proc_id),
            "team_id": int(team_id),
            "disturbance": float(disturbance),
        }
    duration = finish_time - start_time
    team = nowstation.teams[team_id]
    order_stfi[proc_id] = [start_time, finish_time]
    team.stfi[1] = finish_time
    team.order_buffer.append(proc_id)
    team.time_past += duration
    team.timelist[proc_id] = duration
    planned_team_finish[team_id] = finish_time
    planned_orders.add(proc_id)
    finished_orders.add(proc_id)
    air.orders_free = get_available_combo_orders(
        air, planned_orders=planned_orders, finished_orders=finished_orders
    )

    return {
        "valid": True,
        "action_id": int(action_id),
        "proc_idx": int(proc_idx),
        "proc_id": int(proc_id),
        "team_id": int(team_id),
        "start_time": float(start_time),
        "finish_time": float(finish_time),
        "duration": float(duration),
        "disturbance": float(disturbance),
    }


def get_planning_state(env, air, allstation, thispulse, finished_orders, candidates):
    now_time = env.now
    station_id = _station_id_from_time(now_time, thispulse)
    nowstation = allstation[station_id]
    remaino_num = pro_num - len(finished_orders)
    maxdepth = 0
    avgdepth = 0
    maxdeptime = 0
    avgdeptime = 0
    for proc_id in pro_id:
        proc_id = int(proc_id)
        if proc_id in finished_orders:
            continue
        root_depth = dict_postnum[proc_id] + 1
        deptime = dict_posttime[proc_id]
        maxdeptime = max(maxdeptime, deptime)
        maxdepth = max(root_depth, maxdepth)
        avgdepth += root_depth
        avgdeptime += deptime
    avgdepth = avgdepth / remaino_num if remaino_num else 0
    avgdeptime = avgdeptime / remaino_num if remaino_num else 0
    return [
        nowstation.id / 10,
        now_time / 1000,
        len(candidates) / 10,
        remaino_num / 10,
        maxdepth / 10,
        avgdepth / 10,
        maxdeptime / 1000,
        avgdeptime / 1000,
    ], remaino_num == 0


def _pad_or_trim(seq, target_len, pad_value):
    if len(seq) > target_len:
        return seq[:target_len]
    while len(seq) < target_len:
        seq.append(pad_value)
    return seq


def calculate_end2end_final_rewards(
    conf, final_pulse, smoothness_raw, station_times, action_trace, thispulse
):
    base_reward = float(base.get_final_reward(conf, final_pulse, smoothness_raw))
    penalty_weight = float(getattr(conf, "end2end_last_station_penalty_weight", 0.0))
    si_penalty_weight = float(getattr(conf, "end2end_final_si_penalty_weight", 0.0))
    final_si = math.sqrt(max(float(smoothness_raw), 0.0))
    si_penalty = si_penalty_weight * final_si / max(float(final_pulse), 1.0)
    if station_times is not None and len(station_times) > 0:
        last_station_overload = max(0.0, float(station_times[-1]) - float(thispulse)) / max(
            float(thispulse), 1.0
        )
    else:
        last_station_overload = 0.0

    rewards = []
    for item in action_trace:
        reward = base_reward - si_penalty
        if int(item.get("station_id", -1)) == station_num - 1:
            reward -= penalty_weight * last_station_overload
        rewards.append([float(reward)])

    return rewards, {
        "base_reward": base_reward,
        "final_si_penalty": float(si_penalty),
        "final_si_penalty_weight": si_penalty_weight,
        "last_station_overload": float(last_station_overload),
        "last_station_penalty_weight": penalty_weight,
    }


def episode_epsilon(conf, episode_num, evaluate=False):
    if evaluate:
        return 0
    decay_fraction = float(getattr(conf, "end2end_epsilon_decay_fraction", 0.70))
    decay_epoch = max(1, int(conf.n_epochs * decay_fraction))
    if episode_num >= decay_epoch:
        return 0
    return conf.start_epsilon * (1.0 - float(episode_num) / float(decay_epoch))


def generate_episode(
    agents,
    conf,
    pulses,
    thispulse,
    episode_num,
    SI,
    evaluate=False,
    forced_action_trace=None,
    forced_disturbance_trace=None,
    strict_forced_trace=True,
    disable_disturbance=False,
):
    env = simpy.Environment()
    o, u, r, s, o_, s_ = [], [], [], [], [], []
    avail_u, avail_u_, u_onehot, terminate, padded = [], [], [], [], []
    order_mask, order_mask_ = [], []
    combo_pair_features, combo_pair_features_ = [], []
    action_trace = []
    result_holder = {"final_pulse": None, "smoothness_raw": None, "station_times": None, "si": None}
    forced_action_idx = 0

    allstation, _, air = base.reset_env(env, thispulse)

    def production():
        nonlocal forced_action_idx
        last_action = np.zeros((conf.n_agents, conf.n_actions))
        agents.policy.init_hidden(1)
        epsilon = episode_epsilon(conf, episode_num, evaluate)
        done = False
        pending_next_idx = None

        while not done:
            station_id = _station_id_from_time(env.now, thispulse)
            nowstation = allstation[station_id]
            for team in nowstation.teams:
                team.stfi = [nowstation.id * nowstation.pulse, nowstation.id * nowstation.pulse]
                team.order_buffer = []
                team.timelist = {}
            nowstation.orderfinishair = air.order_finish[:]
            nowstation.orderleftair = air.order_left[:]

            planned_orders = set()
            finished_for_planning = set(air.order_finish)
            planned_team_finish = [float(team.stfi[1]) for team in nowstation.teams]
            blocked_teams = set()
            order_stfi = _init_order_stfi(air)
            air.orders_free = get_available_combo_orders(
                air, planned_orders=planned_orders, finished_orders=finished_for_planning
            )
            nowstation.orderfreeair = air.orders_free[:]
            start_candidates = get_available_combo_orders(
                air, planned_orders=planned_orders, finished_orders=finished_for_planning
            )
            start_avail_mask = build_end2end_avail_mask(
                air,
                nowstation,
                planned_orders,
                planned_team_finish,
                thispulse,
                finished_orders=finished_for_planning,
                order_stfi=order_stfi,
                blocked_teams=blocked_teams,
            )
            start_pair_features = build_combo_pair_features(
                air,
                nowstation,
                planned_orders,
                planned_team_finish,
                thispulse,
                finished_orders=finished_for_planning,
                order_stfi=order_stfi,
                blocked_teams=blocked_teams,
                avail_mask=start_avail_mask,
            )
            start_state, start_done = get_planning_state(
                env, air, allstation, thispulse, finished_for_planning, start_candidates
            )
            if pending_next_idx is not None:
                o_[pending_next_idx] = [start_state]
                s_[pending_next_idx] = start_state
                avail_u_[pending_next_idx] = [start_avail_mask]
                order_mask_[pending_next_idx] = agents.policy.get_order_mask_numpy(start_candidates)
                combo_pair_features_[pending_next_idx] = start_pair_features
                terminate[pending_next_idx] = [start_done]
                pending_next_idx = None

            while True:
                candidates = get_available_combo_orders(
                    air, planned_orders=planned_orders, finished_orders=finished_for_planning
                )
                avail_mask = build_end2end_avail_mask(
                    air,
                    nowstation,
                    planned_orders,
                    planned_team_finish,
                    thispulse,
                    finished_orders=finished_for_planning,
                    order_stfi=order_stfi,
                    blocked_teams=blocked_teams,
                )
                if not bool(np.any(avail_mask > 0)):
                    break
                current_pair_features = build_combo_pair_features(
                    air,
                    nowstation,
                    planned_orders,
                    planned_team_finish,
                    thispulse,
                    finished_orders=finished_for_planning,
                    order_stfi=order_stfi,
                    blocked_teams=blocked_teams,
                    avail_mask=avail_mask,
                )

                state, _ = get_planning_state(
                    env, air, allstation, thispulse, finished_for_planning, candidates
                )
                current_order_mask = agents.policy.get_order_mask_numpy(candidates)
                should_record = len(o) < conf.episode_limit
                decision_step_idx = len(action_trace)
                forced_disturbance = None
                if (
                    forced_disturbance_trace is not None
                    and decision_step_idx < len(forced_disturbance_trace)
                ):
                    forced_disturbance = float(forced_disturbance_trace[decision_step_idx])
                if forced_action_trace is not None:
                    if forced_action_idx >= len(forced_action_trace):
                        raise RuntimeError(
                            "forced action trace ended early at step {}".format(forced_action_idx)
                        )
                    action = int(forced_action_trace[forced_action_idx])
                elif should_record:
                    action = agents.choose_action(
                        state,
                        last_action[0],
                        0,
                        avail_mask,
                        epsilon,
                        evaluate,
                        order_candidates=candidates,
                        decision_context={
                            "combo_pair_features_by_action": current_pair_features,
                        },
                    )
                else:
                    action = int(np.nonzero(avail_mask > 0)[0][0])

                plan_record = plan_one_combo_action(
                    nowstation,
                    air,
                    action,
                    planned_orders,
                    planned_team_finish,
                    order_stfi,
                    thispulse,
                    finished_orders=finished_for_planning,
                    blocked_teams=blocked_teams,
                    allow_disturbance=not disable_disturbance,
                    forced_disturbance=forced_disturbance,
                )
                if not plan_record["valid"]:
                    if forced_action_trace is not None and strict_forced_trace:
                        raise RuntimeError(
                            "forced action trace diverged at step {}: action_id={} proc_id={} team_id={} valid={} blocked_team={}".format(
                                forced_action_idx,
                                action,
                                plan_record.get("proc_id"),
                                plan_record.get("team_id"),
                                plan_record.get("valid"),
                                plan_record.get("blocked_team", False),
                            )
                        )
                    if plan_record.get("blocked_team"):
                        continue
                    break
                plan_record["station_id"] = int(station_id)
                plan_record["episode_time"] = float(env.now)
                action_trace.append(plan_record)
                if forced_action_trace is not None:
                    forced_action_idx += 1

                next_candidates = get_available_combo_orders(
                    air, planned_orders=planned_orders, finished_orders=finished_for_planning
                )
                next_avail_mask = build_end2end_avail_mask(
                    air,
                    nowstation,
                    planned_orders,
                    planned_team_finish,
                    thispulse,
                    finished_orders=finished_for_planning,
                    order_stfi=order_stfi,
                    blocked_teams=blocked_teams,
                )
                next_state, next_done = get_planning_state(
                    env, air, allstation, thispulse, finished_for_planning, next_candidates
                )
                next_pair_features = build_combo_pair_features(
                    air,
                    nowstation,
                    planned_orders,
                    planned_team_finish,
                    thispulse,
                    finished_orders=finished_for_planning,
                    order_stfi=order_stfi,
                    blocked_teams=blocked_teams,
                    avail_mask=next_avail_mask,
                )

                action_onehot = np.zeros(conf.n_actions)
                action_onehot[action] = 1.0
                last_action[0] = action_onehot
                if should_record:
                    o.append([state])
                    s.append(state)
                    u.append([[action]])
                    u_onehot.append([action_onehot])
                    avail_u.append([avail_mask])
                    order_mask.append(current_order_mask)
                    combo_pair_features.append(current_pair_features)
                    if bool(np.any(next_avail_mask > 0)) or next_done:
                        avail_u_.append([next_avail_mask])
                        order_mask_.append(agents.policy.get_order_mask_numpy(next_candidates))
                        combo_pair_features_.append(next_pair_features)
                        o_.append([next_state])
                        s_.append(next_state)
                    else:
                        avail_u_.append([np.zeros(conf.n_actions)])
                        order_mask_.append(np.zeros(conf.gnn_node_count))
                        combo_pair_features_.append(
                            np.zeros((conf.n_actions, conf.combo_pair_feature_dim))
                        )
                        o_.append([np.zeros(conf.state_shape).tolist()])
                        s_.append(np.zeros(conf.state_shape).tolist())
                        pending_next_idx = len(o_) - 1
                    r.append([0.0])
                    terminate.append([next_done])
                    padded.append([0.0])

            nowstation.aircraft = air
            for team_id in range(team_num):
                env.process(nowstation.team_process(air, team_id))
            if nowstation.id < station_num - 1:
                yield env.timeout(thispulse)
            else:
                yield env.timeout(thispulse + 1000)
            yield env.timeout(0)
            nowstation.time_past = max(team.time_past for team in nowstation.teams)

            done = len(air.order_finish) == pro_num
            if nowstation.id == station_num - 1:
                final_pulse, smoothness_raw = base.get_pulse(allstation)
                station_times = base.get_station_times(allstation)
                result_holder["final_pulse"] = final_pulse
                result_holder["smoothness_raw"] = smoothness_raw
                result_holder["station_times"] = station_times
                result_holder["si"] = math.sqrt(smoothness_raw)
                pulses.append(final_pulse)
                SI.append(smoothness_raw)
                done = True
                if pending_next_idx is not None:
                    final_state, _ = get_planning_state(
                        env,
                        air,
                        allstation,
                        thispulse,
                        set(air.order_finish),
                        [],
                    )
                    o_[pending_next_idx] = [final_state]
                    s_[pending_next_idx] = final_state
                    avail_u_[pending_next_idx] = [np.zeros(conf.n_actions)]
                    order_mask_[pending_next_idx] = np.zeros(conf.gnn_node_count)
                    combo_pair_features_[pending_next_idx] = np.zeros(
                        (conf.n_actions, conf.combo_pair_feature_dim)
                    )
                    terminate[pending_next_idx] = [True]
                    pending_next_idx = None

    env.process(production())
    if forced_disturbance_trace is not None and forced_action_trace is not None:
        if len(forced_disturbance_trace) != len(forced_action_trace):
            raise ValueError("forced disturbance trace length must match forced action trace")
    env.run(5000)

    transition_len = len(o)
    final_pulse = result_holder["final_pulse"] or 0.0
    smoothness_raw = result_holder["smoothness_raw"] or 0.0
    reward_info = {
        "base_reward": 0.0,
        "final_si_penalty": 0.0,
        "final_si_penalty_weight": float(
            getattr(conf, "end2end_final_si_penalty_weight", 0.0)
        ),
        "last_station_overload": 0.0,
        "last_station_penalty_weight": float(
            getattr(conf, "end2end_last_station_penalty_weight", 0.0)
        ),
    }
    if transition_len and result_holder["final_pulse"] is not None:
        final_rewards, reward_info = calculate_end2end_final_rewards(
            conf,
            final_pulse,
            smoothness_raw,
            result_holder["station_times"] or [],
            action_trace[:transition_len],
            thispulse,
        )
        r = _pad_or_trim(final_rewards, transition_len, [reward_info["base_reward"]])
        terminate[-1] = [True]

    zero_state = np.zeros(conf.state_shape).tolist()
    zero_obs = [zero_state]
    zero_action = np.zeros((conf.n_agents, 1)).tolist()
    zero_onehot = np.zeros((conf.n_agents, conf.n_actions)).tolist()
    zero_avail = np.zeros((conf.n_agents, conf.n_actions)).tolist()
    zero_order_mask = np.zeros(conf.gnn_node_count).tolist()
    zero_pair_features = np.zeros((conf.n_actions, conf.combo_pair_feature_dim), dtype=np.float32)
    final_station_times = result_holder["station_times"] or [0.0] * station_num
    final_si = result_holder["si"] or 0.0

    o = _pad_or_trim(o, conf.episode_limit, zero_obs)
    s = _pad_or_trim(s, conf.episode_limit, zero_state)
    u = _pad_or_trim(u, conf.episode_limit, zero_action)
    r = _pad_or_trim(r, conf.episode_limit, [0.0])
    o_ = _pad_or_trim(o_, conf.episode_limit, zero_obs)
    s_ = _pad_or_trim(s_, conf.episode_limit, zero_state)
    u_onehot = _pad_or_trim(u_onehot, conf.episode_limit, zero_onehot)
    avail_u = _pad_or_trim(avail_u, conf.episode_limit, zero_avail)
    avail_u_ = _pad_or_trim(avail_u_, conf.episode_limit, zero_avail)
    order_mask = _pad_or_trim(order_mask, conf.episode_limit, zero_order_mask)
    order_mask_ = _pad_or_trim(order_mask_, conf.episode_limit, zero_order_mask)
    combo_pair_features = _pad_or_trim(combo_pair_features, conf.episode_limit, zero_pair_features)
    combo_pair_features_ = _pad_or_trim(combo_pair_features_, conf.episode_limit, zero_pair_features)
    padded = _pad_or_trim(padded, conf.episode_limit, [1.0])
    terminate = _pad_or_trim(terminate, conf.episode_limit, [1.0])

    final_station_times_seq = [list(final_station_times) for _ in range(conf.episode_limit)]
    final_si_seq = [[final_si] for _ in range(conf.episode_limit)]
    episode = {
        "o": np.array([o]),
        "s": np.array([s]),
        "u": np.array([u]),
        "r": np.array([r]),
        "o_": np.array([o_]),
        "s_": np.array([s_]),
        "avail_u": np.array([avail_u]),
        "avail_u_": np.array([avail_u_]),
        "u_onehot": np.array([u_onehot]),
        "order_mask": np.array([order_mask]),
        "order_mask_": np.array([order_mask_]),
        "combo_pair_features": np.array([combo_pair_features]),
        "combo_pair_features_": np.array([combo_pair_features_]),
        "final_station_times": np.array([final_station_times_seq]),
        "final_si": np.array([final_si_seq]),
        "padded": np.array([padded]),
        "terminated": np.array([terminate]),
    }
    summary = {
        "final_pulse": float(final_pulse),
        "smoothness_index": float(final_si),
        "station_times": [float(item) for item in final_station_times],
        "actions": action_trace,
        "finished_order_count": int(len(air.order_finish)),
        "final_reward_base": float(reward_info["base_reward"]),
        "final_si_penalty": float(reward_info["final_si_penalty"]),
        "final_si_penalty_weight": float(reward_info["final_si_penalty_weight"]),
        "last_station_overload": float(reward_info["last_station_overload"]),
        "last_station_penalty_weight": float(reward_info["last_station_penalty_weight"]),
        "teacher_forcing": bool(forced_action_trace is not None),
        "forced_action_trace_length": int(len(forced_action_trace or [])),
        "forced_action_trace_consumed": int(forced_action_idx),
    }
    if forced_action_trace is not None and forced_action_idx != len(forced_action_trace):
        raise RuntimeError(
            "forced action trace length mismatch: consumed {} expected {}".format(
                forced_action_idx, len(forced_action_trace)
            )
        )
    return episode, final_pulse, smoothness_raw, summary
