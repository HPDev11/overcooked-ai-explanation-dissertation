from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from overcooked_ai_py.mdp.actions import Action, Direction

Pos = Tuple[int, int]

ACTION_NAME_TO_OBJ = {
    "up": Direction.NORTH,
    "down": Direction.SOUTH,
    "right": Direction.EAST,
    "left": Direction.WEST,
    "stay": Action.STAY,
    "interact": Action.INTERACT,
}
OBJ_TO_ACTION_IDX = {a: i for i, a in enumerate(Action.INDEX_TO_ACTION)}


def action_to_idx(action_obj):
    return OBJ_TO_ACTION_IDX[action_obj]


def action_name(action_obj) -> str:
    if action_obj == Direction.NORTH:
        return "up"
    if action_obj == Direction.SOUTH:
        return "down"
    if action_obj == Direction.EAST:
        return "right"
    if action_obj == Direction.WEST:
        return "left"
    if action_obj == Action.STAY:
        return "stay"
    if action_obj == Action.INTERACT:
        return "interact"
    return str(action_obj)


def held_name(player) -> Optional[str]:
    return player.held_object.name if player.has_object() else None


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def direction_from_to(src: Pos, dst: Pos):
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    candidate = (dx, dy)
    if candidate in Direction.ALL_DIRECTIONS:
        return candidate
    raise ValueError(f"No cardinal direction from {src} to {dst}")


def in_bounds(mdp, pos: Pos) -> bool:
    x, y = pos
    return 0 <= x < mdp.width and 0 <= y < mdp.height


def terrain_at(mdp, pos: Pos) -> Optional[str]:
    if not in_bounds(mdp, pos):
        return None
    return mdp.get_terrain_type_at_pos(pos)


def adjacent_floor_tiles(mdp, target_pos: Pos) -> List[Pos]:
    floors = []
    valid = set(mdp.get_valid_player_positions())
    for direction in Direction.ALL_DIRECTIONS:
        candidate = (target_pos[0] - direction[0], target_pos[1] - direction[1])
        if candidate in valid:
            floors.append(candidate)
    return floors


def nearest_pos(origin: Pos, positions: Sequence[Pos]) -> Optional[Pos]:
    if not positions:
        return None
    return min(positions, key=lambda p: (manhattan(origin, p), p[1], p[0]))


def shared_pot_lane_tiles(mdp) -> Set[Pos]:
    tiles: Set[Pos] = set()
    for pot in mdp.get_pot_locations():
        tiles.update(adjacent_floor_tiles(mdp, pot))
    return tiles


def choose_safe_tile(mdp, state, player_idx: int) -> Pos:
    player = state.players[player_idx]
    valid = list(mdp.get_valid_player_positions())
    dangerous = shared_pot_lane_tiles(mdp)
    candidates = [p for p in valid if p not in dangerous]
    if not candidates:
        candidates = valid
    return min(
        candidates,
        key=lambda p: (
            manhattan(player.position, p),
            min(manhattan(p, q) for q in dangerous) if dangerous else 0,
            p[1],
            p[0],
        ),
    )


def bfs_first_action(
    mdp,
    start: Pos,
    goals: Sequence[Pos],
    blocked: Optional[Set[Pos]] = None,
):
    if not goals:
        return Action.STAY

    valid = set(mdp.get_valid_player_positions())
    goal_set = set(goals)
    blocked = set(blocked or set())
    blocked.discard(start)

    if start in goal_set:
        return Action.STAY

    frontier: List[Pos] = [start]
    parent: Dict[Pos, Optional[Pos]] = {start: None}

    while frontier:
        pos = frontier.pop(0)
        for direction in Direction.ALL_DIRECTIONS:
            nxt = Action.move_in_direction(pos, direction)
            if nxt not in valid:
                continue
            if nxt in blocked and nxt not in goal_set:
                continue
            if nxt in parent:
                continue
            parent[nxt] = pos
            if nxt in goal_set:
                cur = nxt
                while parent[cur] != start and parent[cur] is not None:
                    cur = parent[cur]
                return direction_from_to(start, cur)
            frontier.append(nxt)

    return Action.STAY


def go_and_interact(
    mdp,
    state,
    player_idx: int,
    target_pos: Pos,
    blocked: Optional[Set[Pos]] = None,
):
    player = state.players[player_idx]
    player_pos = player.position

    for floor in adjacent_floor_tiles(mdp, target_pos):
        if floor == player_pos:
            desired_dir = direction_from_to(player_pos, target_pos)
            if player.orientation == desired_dir:
                return Action.INTERACT
            return desired_dir

    return bfs_first_action(
        mdp,
        player_pos,
        adjacent_floor_tiles(mdp, target_pos),
        blocked=blocked,
    )


def parse_partner_intent(explanation_text: Optional[str]) -> str:
    if not explanation_text:
        return "unknown"

    text = explanation_text.lower()

    if "collect dish" in text or "getting a dish" in text or "grab a dish" in text:
        return "collect_dish"
    if "collect soup" in text or "pickup soup" in text or "pick up soup" in text:
        return "collect_soup"
    if "serve soup" in text or "deliver soup" in text or "serving soup" in text:
        return "serve_soup"
    if "clear lane" in text or "give you space" in text or "move to a safe tile" in text:
        return "clear_lane"
    if "wait" in text and "cook" in text:
        return "wait_for_cooking"
    if "deliver onion to pot" in text or "bring onion to pot" in text:
        return "deliver_onion_to_pot"
    if "collect onion" in text or "get onion" in text:
        return "collect_onion"

    return "unknown"


@dataclass
class ScriptedServerAgent:
    player_idx: int = 1

    def _wait_tile(self, mdp, state) -> Pos:
        return choose_safe_tile(mdp, state, self.player_idx)

    def act_action(self, mdp, state):
        player = state.players[self.player_idx]
        hold = held_name(player)

        pot_states = mdp.get_pot_states(state)
        ready_pots = mdp.get_ready_pots(pot_states)
        full_not_cooking = mdp.get_full_but_not_cooking_pots(pot_states)
        cooking_pots = mdp.get_cooking_pots(pot_states)
        serve_stations = mdp.get_serving_locations()
        dish_disps = mdp.get_dish_dispenser_locations()

        blocked = {state.players[1 - self.player_idx].position}

        if hold == "soup":
            target = nearest_pos(player.position, serve_stations)
            return (
                go_and_interact(mdp, state, self.player_idx, target, blocked=blocked),
                "serve_soup",
                "serve_soup",
            )

        if hold == "dish":
            if ready_pots:
                target = nearest_pos(player.position, ready_pots)
                return (
                    go_and_interact(mdp, state, self.player_idx, target, blocked=blocked),
                    "collect_soup",
                    "pickup_soup",
                )

            if full_not_cooking:
                wait_tile = self._wait_tile(mdp, state)
                action = bfs_first_action(
                    mdp,
                    player.position,
                    [wait_tile],
                    blocked=blocked,
                )
                return action, "wait_for_cooking", "clear_lane"

            if cooking_pots:
                wait_tile = self._wait_tile(mdp, state)
                action = bfs_first_action(
                    mdp,
                    player.position,
                    [wait_tile],
                    blocked=blocked,
                )
                return action, "wait_for_cooking", "wait"

            wait_tile = self._wait_tile(mdp, state)
            action = bfs_first_action(
                mdp,
                player.position,
                [wait_tile],
                blocked=blocked,
            )
            return action, "wait_for_pot", "wait"

        target = nearest_pos(player.position, dish_disps)
        return (
            go_and_interact(mdp, state, self.player_idx, target, blocked=blocked),
            "collect_dish",
            "pickup_dish",
        )

    def act(self, mdp, state) -> int:
        action_obj, _, _ = self.act_action(mdp, state)
        return action_to_idx(action_obj)


def render_server_explanation(goal_label: str, action_label: str, mode: str = "both") -> str:
    goal_text = {
        "collect_dish": "I am collecting a dish.",
        "collect_soup": "I am collecting soup from the pot.",
        "serve_soup": "I am serving soup now.",
        "wait_for_cooking": "I am waiting for cooking to start and giving you space.",
        "wait_for_pot": "I am waiting for the pot to become available.",
    }.get(goal_label, "I am acting to help the team.")

    action_text = {
        "pickup_dish": "My next action is to grab a dish.",
        "pickup_soup": "My next action is to pick up soup.",
        "serve_soup": "My next action is to deliver soup.",
        "clear_lane": "My next action is to move to a safe tile and clear the lane.",
        "wait": "My next action is to wait out of the way.",
    }.get(action_label, "My next action supports that goal.")

    if mode == "goal":
        return goal_text
    if mode == "action":
        return action_text
    return f"{goal_text} {action_text}"


@dataclass
class HumanSim:
    player_idx: int = 0

    # noexp: deliberately less coordinated / more dithery
    noexp_wait_prob: float = 0.18
    noexp_detour_prob: float = 0.18
    noexp_ignore_lane_prob: float = 0.80

    # exp: mostly stable, but not perfectly deterministic
    exp_wait_prob: float = 0.02

    def _maybe_noexp_noise(self, mdp, state, base_action, rng: random.Random):
        player = state.players[self.player_idx]

        if base_action == Action.INTERACT:
            return base_action

        if rng.random() < self.noexp_wait_prob:
            return Action.STAY

        if base_action in Direction.ALL_DIRECTIONS and rng.random() < self.noexp_detour_prob:
            valid_dirs = []
            valid_positions = set(mdp.get_valid_player_positions())
            for direction in Direction.ALL_DIRECTIONS:
                nxt = Action.move_in_direction(player.position, direction)
                if nxt in valid_positions:
                    valid_dirs.append(direction)
            if valid_dirs:
                return rng.choice(valid_dirs)

        return base_action

    def _maybe_exp_noise(self, mdp, state, base_action, rng: random.Random):
        if base_action == Action.INTERACT:
            return base_action

        if rng.random() < self.exp_wait_prob:
            return Action.STAY

        return base_action

    def _chef_goal_action(self, mdp, state, partner_intent: str, explanation_visible: bool):
        player = state.players[self.player_idx]
        partner = state.players[1 - self.player_idx]
        hold = held_name(player)
        blocked = {partner.position}

        pot_states = mdp.get_pot_states(state)
        ready_pots = mdp.get_ready_pots(pot_states)
        cooking_pots = mdp.get_cooking_pots(pot_states)
        full_not_cooking = mdp.get_full_but_not_cooking_pots(pot_states)
        partial_pots = mdp.get_partially_full_pots(pot_states)
        empty_pots = mdp.get_empty_pots(pot_states)
        onion_disps = mdp.get_onion_dispenser_locations()
        serve_stations = mdp.get_serving_locations()

        fillable_pots = list(partial_pots) + list(empty_pots)
        target_fill_pot = nearest_pos(player.position, fillable_pots)
        target_onion = nearest_pos(player.position, onion_disps)
        target_safe = choose_safe_tile(mdp, state, self.player_idx)
        lane_tiles = shared_pot_lane_tiles(mdp)
        on_lane = player.position in lane_tiles

        partner_needs_lane = partner_intent in {
            "collect_dish",
            "collect_soup",
            "serve_soup",
            "clear_lane",
            "wait_for_cooking",
        }

        if hold == "soup":
            target = nearest_pos(player.position, serve_stations)
            return (
                go_and_interact(mdp, state, self.player_idx, target, blocked=blocked),
                "serve_soup",
            )

        if full_not_cooking and hold is None:
            target = nearest_pos(player.position, full_not_cooking)
            return (
                go_and_interact(mdp, state, self.player_idx, target, blocked=blocked),
                "kickstart_cooking",
            )

        if explanation_visible and partner_needs_lane and on_lane:
            return (
                bfs_first_action(mdp, player.position, [target_safe], blocked=blocked),
                "clear_lane",
            )

        if hold == "onion":
            if target_fill_pot is not None:
                return (
                    go_and_interact(mdp, state, self.player_idx, target_fill_pot, blocked=blocked),
                    "deliver_onion_to_pot",
                )
            return (
                bfs_first_action(mdp, player.position, [target_safe], blocked=blocked),
                "wait_out_of_way",
            )

        if hold == "dish" and ready_pots:
            target = nearest_pos(player.position, ready_pots)
            return (
                go_and_interact(mdp, state, self.player_idx, target, blocked=blocked),
                "collect_soup",
            )

        if target_fill_pot is not None:
            return (
                go_and_interact(mdp, state, self.player_idx, target_onion, blocked=blocked),
                "collect_onion",
            )

        if ready_pots or cooking_pots or full_not_cooking:
            return (
                bfs_first_action(mdp, player.position, [target_safe], blocked=blocked),
                "wait_out_of_way",
            )

        return (
            bfs_first_action(mdp, player.position, [target_safe], blocked=blocked),
            "wait_out_of_way",
        )

    def act_action(self, mdp, state, explanation_text: Optional[str], rng: random.Random):
        partner_intent = parse_partner_intent(explanation_text)
        explanation_visible = bool(explanation_text)
        base_action, goal_label = self._chef_goal_action(
            mdp,
            state,
            partner_intent,
            explanation_visible,
        )

        player = state.players[self.player_idx]
        on_lane = player.position in shared_pot_lane_tiles(mdp)

        if (not explanation_visible) and on_lane and partner_intent in {
            "collect_soup",
            "collect_dish",
            "wait_for_cooking",
        }:
            if rng.random() < self.noexp_ignore_lane_prob:
                return self._maybe_noexp_noise(mdp, state, base_action, rng), goal_label
            return (
                bfs_first_action(
                    mdp,
                    player.position,
                    [choose_safe_tile(mdp, state, self.player_idx)],
                ),
                "clear_lane",
            )

        if not explanation_visible:
            return self._maybe_noexp_noise(mdp, state, base_action, rng), goal_label

        return self._maybe_exp_noise(mdp, state, base_action, rng), goal_label

    def act(self, mdp, state, explanation_text: Optional[str], rng: random.Random) -> int:
        action_obj, _ = self.act_action(mdp, state, explanation_text, rng)
        return action_to_idx(action_obj)