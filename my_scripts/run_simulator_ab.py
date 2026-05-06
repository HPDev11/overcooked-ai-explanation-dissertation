import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from human_simulator_agent import (
    HumanSim,
    ScriptedServerAgent,
    action_name,
    action_to_idx,
    held_name,
    render_server_explanation,
)

Pos = Tuple[int, int]

KEY_EVENTS = {
    "picked_onion",
    "placed_or_dropped_onion",
    "picked_dish",
    "placed_or_dropped_dish",
    "picked_soup",
    "delivered_or_dropped_soup",
}

ACTION_NAMES = {i: action_name(a) for i, a in enumerate(Action.INDEX_TO_ACTION)}

CSV_COLUMNS = [
    "layout",
    "condition",
    "mode",
    "seed",
    "horizon",
    "goal_change_only",
    "event_trigger",
    "t",
    "a0",
    "a1",
    "a0_name",
    "a1_name",
    "p0",
    "p1",
    "p0_next",
    "p1_next",
    "held0_before",
    "held1_before",
    "held0_after",
    "held1_after",
    "event0",
    "event1",
    "moved0",
    "moved1",
    "stay0",
    "stay1",
    "stuck0",
    "stuck1",
    "blocked_wall0",
    "blocked_wall1",
    "blocked_partner0",
    "blocked_partner1",
    "collision",
    "objs_before",
    "objs_after",
    "obj_added",
    "obj_removed",
    "handoff_drop0",
    "handoff_drop1",
    "handoff_pickup0",
    "handoff_pickup1",
    "handoff_total",
    "goal0",
    "goal1",
    "exp0",
    "exp1",
    "r_game",
    "r_shaped_total",
    "soup_delivered_step",
    "productive0_step",
    "productive1_step",
    "activity0_step",
    "activity1_step",
    "interactivity0_step",
    "interactivity1_step",
    "adjacent_step",
    "plate_pickup_total_step",
    "plate_pickup_while_cooking_step",
    "pot_empty_n",
    "pot_cooking_n",
    "pot_ready_n",
    "pot_total_n",
]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run explanation vs no-explanation simulator A/B experiments."
    )
    ap.add_argument("--layout", type=str, default="cramped_room")
    ap.add_argument("--horizon", type=int, default=800)
    ap.add_argument("--n_seeds", type=int, default=6)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--mode", type=str, default="both", choices=["action", "goal", "both"])
    ap.add_argument("--goal_change_only", action="store_true")
    ap.add_argument("--event_trigger", action="store_true")
    ap.add_argument("--logs_dir", type=str, default="logs")
    return ap.parse_args()


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def is_adjacent(a: Pos, b: Pos) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def candidate_pos(_mdp, pos: Pos, action_obj):
    if action_obj in Direction.ALL_DIRECTIONS:
        return Action.move_in_direction(pos, action_obj)
    return pos


def attempt_valid_move(mdp, pos: Pos, action_obj) -> bool:
    if action_obj not in Direction.ALL_DIRECTIONS:
        return False
    nxt = Action.move_in_direction(pos, action_obj)
    return nxt in set(mdp.get_valid_player_positions())


def obj_set(state):
    objs = state.to_dict().get("objects", [])
    return set((o.get("name"), tuple(o.get("position"))) for o in objs)


def infer_event(held_before: Optional[str], held_after: Optional[str]) -> Optional[str]:
    if held_before == held_after:
        return None

    if held_before is None and held_after is not None:
        if held_after == "onion":
            return "picked_onion"
        if held_after == "dish":
            return "picked_dish"
        if "soup" in str(held_after).lower():
            return "picked_soup"
        return f"picked_{held_after}"

    if held_before is not None and held_after is None:
        if held_before == "onion":
            return "placed_or_dropped_onion"
        if held_before == "dish":
            return "placed_or_dropped_dish"
        if "soup" in str(held_before).lower():
            return "delivered_or_dropped_soup"
        return f"placed_or_dropped_{held_before}"

    return f"swapped_{held_before}_to_{held_after}"


def get_pot_state_counts(mdp, state):
    pot_states = mdp.get_pot_states(state)
    ready = list(mdp.get_ready_pots(pot_states))
    cooking = list(mdp.get_cooking_pots(pot_states))
    full_not_cooking = list(mdp.get_full_but_not_cooking_pots(pot_states))
    partial = list(mdp.get_partially_full_pots(pot_states))
    empty = list(mdp.get_empty_pots(pot_states))
    total = len(mdp.get_pot_locations())
    return {
        "ready": ready,
        "cooking": cooking,
        "full_not_cooking": full_not_cooking,
        "partial": partial,
        "empty": empty,
        "total": total,
    }


def summarize_rows(
    rows: List[Dict[str, object]],
    layout: str,
    condition: str,
    mode: str,
    seed: int,
    horizon: int,
    goal_change_only: bool,
    event_trigger: bool,
):
    if not rows:
        return {
            "layout": layout,
            "condition": condition,
            "mode": mode,
            "seed": seed,
            "horizon": horizon,
            "goal_change_only": goal_change_only,
            "event_trigger": event_trigger,
            "timesteps": 0,
        }

    T = len(rows)

    def fsum(name: str) -> float:
        return sum(float(r.get(name, 0.0)) for r in rows)

    def isum(name: str) -> int:
        return sum(int(r.get(name, 0)) for r in rows)

    total_game_reward = fsum("r_game")
    total_shaped_reward = fsum("r_shaped_total")

    collisions = isum("collision")
    blocked_partner0 = isum("blocked_partner0")
    blocked_partner1 = isum("blocked_partner1")
    blocked_partner_total = blocked_partner0 + blocked_partner1

    blocked_wall0 = isum("blocked_wall0")
    blocked_wall1 = isum("blocked_wall1")
    blocked_wall_total = blocked_wall0 + blocked_wall1

    idle0 = [1 if int(r.get("stay0", 0)) or int(r.get("stuck0", 0)) else 0 for r in rows]
    idle1 = [1 if int(r.get("stay1", 0)) or int(r.get("stuck1", 0)) else 0 for r in rows]
    idle_rate0 = sum(idle0) / T
    idle_rate1 = sum(idle1) / T
    idle_rate_avg = (idle_rate0 + idle_rate1) / 2.0

    handoff_total = int(rows[-1].get("handoff_total", 0))

    objs_before_avg = fsum("objs_before") / T
    objs_after_avg = fsum("objs_after") / T

    exp0_count = sum(1 for r in rows if str(r.get("exp0", "")).strip())
    exp1_count = sum(1 for r in rows if str(r.get("exp1", "")).strip())
    exp_any_count = sum(
        1 for r in rows if str(r.get("exp0", "")).strip() or str(r.get("exp1", "")).strip()
    )
    exp0_rate = exp0_count / T
    exp1_rate = exp1_count / T
    exp_any_rate = exp_any_count / T

    soups_delivered = isum("soup_delivered_step")
    delivery_timesteps = [int(r["t"]) for r in rows if int(r.get("soup_delivered_step", 0)) == 1]
    time_to_first_delivery = delivery_timesteps[0] if delivery_timesteps else horizon

    if len(delivery_timesteps) >= 2:
        delivery_intervals = [
            delivery_timesteps[i] - delivery_timesteps[i - 1]
            for i in range(1, len(delivery_timesteps))
        ]
        mean_delivery_interval = statistics.mean(delivery_intervals)
    else:
        mean_delivery_interval = float(horizon)

    productive_actions_0 = isum("productive0_step")
    productive_actions_1 = isum("productive1_step")
    productive_actions_total = productive_actions_0 + productive_actions_1

    contribution_ratio_0 = (
        productive_actions_0 / productive_actions_total if productive_actions_total > 0 else 0.0
    )
    contribution_ratio_1 = (
        productive_actions_1 / productive_actions_total if productive_actions_total > 0 else 0.0
    )

    activity_count_0 = isum("activity0_step")
    activity_count_1 = isum("activity1_step")
    activity_0 = activity_count_0 / T
    activity_1 = activity_count_1 / T

    interactivity_count_0 = isum("interactivity0_step")
    interactivity_count_1 = isum("interactivity1_step")
    interactivity_0 = interactivity_count_0 / T
    interactivity_1 = interactivity_count_1 / T

    efficiency_0 = total_game_reward / activity_0 if activity_0 > 0 else 0.0
    efficiency_1 = total_game_reward / activity_1 if activity_1 > 0 else 0.0

    adjacency_rate = isum("adjacent_step") / T

    plate_pickup_total = isum("plate_pickup_total_step")
    plate_pickup_while_cooking_count = isum("plate_pickup_while_cooking_step")
    plate_pickup_while_cooking_rate = (
        plate_pickup_while_cooking_count / plate_pickup_total if plate_pickup_total > 0 else 0.0
    )

    total_pot_obs = isum("pot_total_n")
    pot_empty_rate = fsum("pot_empty_n") / total_pot_obs if total_pot_obs > 0 else 0.0
    pot_cooking_rate = fsum("pot_cooking_n") / total_pot_obs if total_pot_obs > 0 else 0.0
    pot_ready_wait_rate = fsum("pot_ready_n") / total_pot_obs if total_pot_obs > 0 else 0.0

    return {
        "layout": layout,
        "condition": condition,
        "mode": mode,
        "seed": seed,
        "horizon": horizon,
        "goal_change_only": goal_change_only,
        "event_trigger": event_trigger,
        "timesteps": T,
        "total_game_reward": total_game_reward,
        "total_shaped_reward": total_shaped_reward,
        "collisions": collisions,
        "blocked_partner_total": blocked_partner_total,
        "blocked_partner0": blocked_partner0,
        "blocked_partner1": blocked_partner1,
        "blocked_wall_total": blocked_wall_total,
        "blocked_wall0": blocked_wall0,
        "blocked_wall1": blocked_wall1,
        "idle_rate0": idle_rate0,
        "idle_rate1": idle_rate1,
        "idle_rate_avg": idle_rate_avg,
        "handoff_total": handoff_total,
        "objs_before_avg": objs_before_avg,
        "objs_after_avg": objs_after_avg,
        "exp0_count": exp0_count,
        "exp1_count": exp1_count,
        "exp_any_count": exp_any_count,
        "exp0_rate": exp0_rate,
        "exp1_rate": exp1_rate,
        "exp_any_rate": exp_any_rate,
        "soups_delivered": soups_delivered,
        "time_to_first_delivery": time_to_first_delivery,
        "mean_delivery_interval": mean_delivery_interval,
        "productive_actions_0": productive_actions_0,
        "productive_actions_1": productive_actions_1,
        "productive_actions_total": productive_actions_total,
        "contribution_ratio_0": contribution_ratio_0,
        "contribution_ratio_1": contribution_ratio_1,
        "activity_count_0": activity_count_0,
        "activity_count_1": activity_count_1,
        "activity_0": activity_0,
        "activity_1": activity_1,
        "interactivity_count_0": interactivity_count_0,
        "interactivity_count_1": interactivity_count_1,
        "interactivity_0": interactivity_0,
        "interactivity_1": interactivity_1,
        "efficiency_0": efficiency_0,
        "efficiency_1": efficiency_1,
        "adjacency_rate": adjacency_rate,
        "plate_pickup_total": plate_pickup_total,
        "plate_pickup_while_cooking_count": plate_pickup_while_cooking_count,
        "plate_pickup_while_cooking_rate": plate_pickup_while_cooking_rate,
        "pot_empty_rate": pot_empty_rate,
        "pot_cooking_rate": pot_cooking_rate,
        "pot_ready_wait_rate": pot_ready_wait_rate,
    }


def run_one_condition(
    layout: str,
    condition: str,
    mode: str,
    seed: int,
    horizon: int,
    goal_change_only: bool,
    event_trigger: bool,
    logs_dir: str,
):
    ensure_dir(logs_dir)

    rng = random.Random(seed)

    mdp = OvercookedGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon)
    env.reset()

    human_sim = HumanSim(player_idx=0)
    server_agent = ScriptedServerAgent(player_idx=1)

    prev_goal1 = None
    prev_event0 = None
    prev_event1 = None
    prev_collision = 0
    prev_handoff_pickup0 = 0
    prev_handoff_pickup1 = 0
    prev_handoff_drop0 = 0
    prev_handoff_drop1 = 0

    rows: List[Dict[str, object]] = []
    done = False
    t = 0
    handoff_total = 0

    stem = f"sim_{layout}_{condition}_{mode}_{seed}"
    csv_path = Path(logs_dir) / f"{stem}.csv"
    summary_path = Path(logs_dir) / f"{stem}_summary.json"

    while not done:
        state = env.state
        objs_b = obj_set(state)

        p0 = state.players[0].position
        p1 = state.players[1].position

        held0_b = held_name(state.players[0])
        held1_b = held_name(state.players[1])

        pot_before = get_pot_state_counts(mdp, state)

        a1_obj, goal1, action_label1 = server_agent.act_action(mdp, state)

        explanation_text = None
        exp1 = ""
        if condition == "exp":
            emit = True
            if goal_change_only:
                emit = prev_goal1 is None or goal1 != prev_goal1

            if event_trigger:
                if (
                    prev_event0 in KEY_EVENTS
                    or prev_event1 in KEY_EVENTS
                    or prev_collision == 1
                    or prev_handoff_pickup0 == 1
                    or prev_handoff_pickup1 == 1
                    or prev_handoff_drop0 == 1
                    or prev_handoff_drop1 == 1
                ):
                    emit = True

            if emit:
                explanation_text = render_server_explanation(goal1, action_label1, mode=mode)
                exp1 = explanation_text

        a0_obj, goal0 = human_sim.act_action(
            mdp=mdp,
            state=state,
            explanation_text=explanation_text,
            rng=rng,
        )

        a0 = action_to_idx(a0_obj)
        a1 = action_to_idx(a1_obj)

        cand0 = candidate_pos(mdp, p0, a0_obj)
        cand1 = candidate_pos(mdp, p1, a1_obj)

        attempt0 = int(attempt_valid_move(mdp, p0, a0_obj))
        attempt1 = int(attempt_valid_move(mdp, p1, a1_obj))

        blocked_wall0 = int(a0_obj in Direction.ALL_DIRECTIONS and not attempt0)
        blocked_wall1 = int(a1_obj in Direction.ALL_DIRECTIONS and not attempt1)

        same_target = attempt0 and attempt1 and (cand0 == cand1)
        swap = attempt0 and attempt1 and (cand0 == p1 and cand1 == p0)
        collision = int(same_target or swap)

        joint_action = (a0_obj, a1_obj)
        _, r_game, done, info = env.step(joint_action)

        shaped = sum(info.get("shaped_r_by_agent", []))

        state2 = env.state
        objs_a = obj_set(state2)
        added = objs_a - objs_b
        removed = objs_b - objs_a

        p0n = state2.players[0].position
        p1n = state2.players[1].position

        held0_a = held_name(state2.players[0])
        held1_a = held_name(state2.players[1])

        event0 = infer_event(held0_b, held0_a)
        event1 = infer_event(held1_b, held1_a)

        moved0 = int(p0n != p0)
        moved1 = int(p1n != p1)

        stay0 = int(a0_obj == Action.STAY)
        stay1 = int(a1_obj == Action.STAY)

        stuck0 = int(attempt0 and not moved0)
        stuck1 = int(attempt1 and not moved1)

        blocked_partner0 = int(attempt0 and not moved0 and (collision == 1 or cand0 == p1))
        blocked_partner1 = int(attempt1 and not moved1 and (collision == 1 or cand1 == p0))

        handoff_drop0 = int(held0_b is not None and held0_a is None and len(added) > 0)
        handoff_drop1 = int(held1_b is not None and held1_a is None and len(added) > 0)
        handoff_pickup0 = int(held0_b is None and held0_a is not None and len(removed) > 0)
        handoff_pickup1 = int(held1_b is None and held1_a is not None and len(removed) > 0)
        handoff_total += handoff_pickup0 + handoff_pickup1

        pot_after = get_pot_state_counts(mdp, state2)
        valid_positions = set(mdp.get_valid_player_positions())
        pot_adj_tiles = set()
        for pot in mdp.get_pot_locations():
            for d in Direction.ALL_DIRECTIONS:
                adj = (pot[0] - d[0], pot[1] - d[1])
                if adj in valid_positions:
                    pot_adj_tiles.add(adj)

        pot_add_success0 = int(
            held0_b == "onion"
            and held0_a is None
            and a0_obj == Action.INTERACT
            and p0 in pot_adj_tiles
        )
        pot_add_success1 = int(
            held1_b == "onion"
            and held1_a is None
            and a1_obj == Action.INTERACT
            and p1 in pot_adj_tiles
        )

        cooking_started = len(pot_after["cooking"]) > len(pot_before["cooking"])

        kickstart0 = int(
            cooking_started
            and len(pot_before["full_not_cooking"]) > 0
            and held0_b is None
            and a0_obj == Action.INTERACT
            and p0 in pot_adj_tiles
        )
        kickstart1 = int(
            cooking_started
            and len(pot_before["full_not_cooking"]) > 0
            and held1_b is None
            and a1_obj == Action.INTERACT
            and p1 in pot_adj_tiles
        )

        delivery0 = int(r_game > 0 and held0_b is not None and "soup" in held0_b and held0_a is None)
        delivery1 = int(r_game > 0 and held1_b is not None and "soup" in held1_b and held1_a is None)

        productive0_step = 0
        productive1_step = 0

        if event0 == "picked_onion":
            productive0_step += 1
        if event1 == "picked_onion":
            productive1_step += 1

        productive0_step += pot_add_success0
        productive1_step += pot_add_success1

        if event0 == "picked_dish":
            productive0_step += 1
        if event1 == "picked_dish":
            productive1_step += 1

        if event0 == "picked_soup":
            productive0_step += 1
        if event1 == "picked_soup":
            productive1_step += 1

        productive0_step += delivery0 + kickstart0
        productive1_step += delivery1 + kickstart1

        soup_delivered_step = int(r_game > 0)

        activity0_step = int(a0_obj != Action.STAY)
        activity1_step = int(a1_obj != Action.STAY)

        interactivity0_step = int(a0_obj == Action.INTERACT)
        interactivity1_step = int(a1_obj == Action.INTERACT)

        adjacent_step = int(is_adjacent(p0, p1))

        plate_pickup_total_step = int(event0 == "picked_dish") + int(event1 == "picked_dish")
        plate_pickup_while_cooking_step = 0
        if event0 == "picked_dish" and (len(pot_before["cooking"]) > 0 or len(pot_before["ready"]) > 0):
            plate_pickup_while_cooking_step += 1
        if event1 == "picked_dish" and (len(pot_before["cooking"]) > 0 or len(pot_before["ready"]) > 0):
            plate_pickup_while_cooking_step += 1

        row = {
            "layout": layout,
            "condition": condition,
            "mode": mode,
            "seed": seed,
            "horizon": horizon,
            "goal_change_only": goal_change_only,
            "event_trigger": event_trigger,
            "t": t,
            "a0": a0,
            "a1": a1,
            "a0_name": ACTION_NAMES[a0],
            "a1_name": ACTION_NAMES[a1],
            "p0": str(p0),
            "p1": str(p1),
            "p0_next": str(p0n),
            "p1_next": str(p1n),
            "held0_before": held0_b,
            "held1_before": held1_b,
            "held0_after": held0_a,
            "held1_after": held1_a,
            "event0": event0 or "",
            "event1": event1 or "",
            "moved0": moved0,
            "moved1": moved1,
            "stay0": stay0,
            "stay1": stay1,
            "stuck0": stuck0,
            "stuck1": stuck1,
            "blocked_wall0": blocked_wall0,
            "blocked_wall1": blocked_wall1,
            "blocked_partner0": blocked_partner0,
            "blocked_partner1": blocked_partner1,
            "collision": collision,
            "objs_before": len(objs_b),
            "objs_after": len(objs_a),
            "obj_added": len(added),
            "obj_removed": len(removed),
            "handoff_drop0": handoff_drop0,
            "handoff_drop1": handoff_drop1,
            "handoff_pickup0": handoff_pickup0,
            "handoff_pickup1": handoff_pickup1,
            "handoff_total": handoff_total,
            "goal0": goal0,
            "goal1": goal1,
            "exp0": "",
            "exp1": exp1,
            "r_game": float(r_game),
            "r_shaped_total": float(shaped),
            "soup_delivered_step": soup_delivered_step,
            "productive0_step": productive0_step,
            "productive1_step": productive1_step,
            "activity0_step": activity0_step,
            "activity1_step": activity1_step,
            "interactivity0_step": interactivity0_step,
            "interactivity1_step": interactivity1_step,
            "adjacent_step": adjacent_step,
            "plate_pickup_total_step": plate_pickup_total_step,
            "plate_pickup_while_cooking_step": plate_pickup_while_cooking_step,
            "pot_empty_n": len(pot_before["empty"]),
            "pot_cooking_n": len(pot_before["cooking"]),
            "pot_ready_n": len(pot_before["ready"]),
            "pot_total_n": pot_before["total"],
        }
        rows.append(row)

        prev_goal1 = goal1
        prev_event0 = event0
        prev_event1 = event1
        prev_collision = collision
        prev_handoff_pickup0 = handoff_pickup0
        prev_handoff_pickup1 = handoff_pickup1
        prev_handoff_drop0 = handoff_drop0
        prev_handoff_drop1 = handoff_drop1

        t += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_rows(
        rows=rows,
        layout=layout,
        condition=condition,
        mode=mode,
        seed=seed,
        horizon=horizon,
        goal_change_only=goal_change_only,
        event_trigger=event_trigger,
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")


def main():
    args = parse_args()

    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        for condition in ["noexp", "exp"]:
            run_one_condition(
                layout=args.layout,
                condition=condition,
                mode=args.mode,
                seed=seed,
                horizon=args.horizon,
                goal_change_only=args.goal_change_only,
                event_trigger=args.event_trigger,
                logs_dir=args.logs_dir,
            )


if __name__ == "__main__":
    main()
