import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import pygame

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action


# ============================================================
# Actions (0 up, 1 down, 2 right, 3 left, 4 stay, 5 interact)
# ============================================================
ACTION_NAMES = {0: "up", 1: "down", 2: "right", 3: "left", 4: "stay", 5: "interact"}
ACTION_DELTAS = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}
MOVE_ACTIONS = {0, 1, 2, 3}
STAY = 4
INTERACT = 5

# Terrain chars that are not walkable floor
NON_WALKABLE = {"X", "O", "D", "P", "S", "T"}

KICK_STEPS = 6
INSTR_SEC = 10.0


# ============================================================
# CLI
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", type=str, required=True, help="e.g. P01")
    ap.add_argument("--condition", type=str, default="noexp", choices=["noexp", "exp"])
    ap.add_argument("--layout", type=str, default="cramped_room")
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--fps", type=int, default=8, help="steps per second (play speed)")
    ap.add_argument("--explain_mode", type=str, default="both", choices=["action", "goal", "both"])
    ap.add_argument("--goal_change_only", action="store_true", help="rate limit explanations by goal change")
    ap.add_argument("--event_trigger", action="store_true", help="also emit explanations on events")
    ap.add_argument("--out_dir", type=str, default="logs")
    ap.add_argument("--run_id", type=str, default="", help="e.g. PRACTICE, A1, A2, B1, B2")
    return ap.parse_args()


# ============================================================
# Small helpers
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_bounds(mdp, pos):
    x, y = pos
    return 0 <= y < len(mdp.terrain_mtx) and 0 <= x < len(mdp.terrain_mtx[0])


def is_walkable(mdp, pos):
    if not in_bounds(mdp, pos):
        return False
    x, y = pos
    return mdp.terrain_mtx[y][x] not in NON_WALKABLE


def find_tiles(mdp, ch: str) -> List[Tuple[int, int]]:
    tiles = []
    for y, row in enumerate(mdp.terrain_mtx):
        for x, c in enumerate(row):
            if c == ch:
                tiles.append((x, y))
    return tiles


def adjacent_floor_tiles(mdp, target_pos) -> List[Tuple[int, int]]:
    x, y = target_pos
    res = []
    for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
        p = (x + dx, y + dy)
        if is_walkable(mdp, p):
            res.append(p)
    return res


def choose_station_tile(mdp, tiles: List[Tuple[int, int]], prefer_near: Optional[Tuple[int, int]] = None):
    """Pick a station tile with at least one adjacent walkable tile."""
    valid = []
    for t in tiles:
        adj = adjacent_floor_tiles(mdp, t)
        if adj:
            valid.append((t, adj))
    if not valid:
        raise RuntimeError(f"No usable station tile among: {tiles}")

    if prefer_near is None:
        return valid[0][0], valid[0][1]

    valid.sort(key=lambda pair: manhattan(pair[0], prefer_near))
    return valid[0][0], valid[0][1]


def held_name(player) -> Optional[str]:
    obj = getattr(player, "held_object", None)
    if obj is None:
        return None
    return getattr(obj, "name", None) or getattr(obj, "obj_type", None) or str(obj)


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


def is_adjacent(a, b) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def orient_action(from_pos, to_pos) -> Optional[int]:
    fx, fy = from_pos
    tx, ty = to_pos
    dx, dy = tx - fx, ty - fy
    for a, (adx, ady) in ACTION_DELTAS.items():
        if (adx, ady) == (dx, dy):
            return a
    return None


# ============================================================
# Make SPACE forgiving for HUMAN (fix: onions not going in pot)
# ============================================================
def human_space_autoorient(state, p0, held0, a0, onion_tile, pot_tile):
    """
    If human presses SPACE but isn't facing the relevant station, do a one-step
    "bump" move to set orientation automatically.
    """
    if a0 != INTERACT:
        return a0

    # Decide which station human intends to interact with
    target = None
    if held0 == "onion" and is_adjacent(p0, pot_tile):
        target = pot_tile
    elif held0 is None and is_adjacent(p0, onion_tile):
        target = onion_tile

    if target is None:
        return a0

    desired = orient_action(p0, target)
    if desired is None:
        return a0

    ori = state.to_dict()["players"][0]["orientation"]
    want_vec = ACTION_DELTAS[desired]
    if tuple(ori) != tuple(want_vec):
        return desired  # bump to set orientation

    return a0


# ============================================================
# Counter occupancy helpers (dish stash fix)
# ============================================================
def state_objects_dict(state) -> dict:
    objs = getattr(state, "objects", None)
    return objs if isinstance(objs, dict) else {}


def obj_name(obj) -> str:
    return str(getattr(obj, "name", None) or getattr(obj, "obj_type", None) or obj).lower()


def choose_empty_counter(mdp, state, counter_tiles: List[Tuple[int, int]], prefer_near: Tuple[int, int]):
    objs = state_objects_dict(state)
    candidates = []
    for ct in counter_tiles:
        if ct in objs:
            continue
        adj = adjacent_floor_tiles(mdp, ct)
        if not adj:
            continue
        candidates.append((ct, adj))

    if not candidates:
        return None, None

    candidates.sort(key=lambda pair: manhattan(pair[0], prefer_near))
    return candidates[0][0], candidates[0][1]


def counter_has_dish(state, counter_pos: Tuple[int, int]) -> bool:
    objs = state_objects_dict(state)
    obj = objs.get(counter_pos)
    if obj is None:
        return False
    return "dish" in obj_name(obj)


# ============================================================
# Simple BFS for the AI (server teammate)
# ============================================================
from collections import deque


def bfs_first_step(mdp, start: Tuple[int, int], goals: set) -> int:
    if not goals or start in goals:
        return STAY

    q = deque([start])
    parent = {start: None}
    parent_action = {start: None}

    while q:
        cur = q.popleft()
        for a, (dx, dy) in ACTION_DELTAS.items():
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in parent:
                continue
            if not is_walkable(mdp, nxt):
                continue
            parent[nxt] = cur
            parent_action[nxt] = a
            if nxt in goals:
                q.clear()
                break
            q.append(nxt)

    reached = None
    for g in goals:
        if g in parent:
            reached = g
            break
    if reached is None:
        return STAY

    cur = reached
    while parent[cur] != start and parent[cur] is not None:
        cur = parent[cur]
    return parent_action[cur]


def go_and_interact(mdp, state, agent_idx: int, station_pos, station_adj) -> int:
    my_pos = state.players[agent_idx].position

    if my_pos not in station_adj:
        return bfs_first_step(mdp, my_pos, set(station_adj))

    if is_adjacent(my_pos, station_pos):
        desired = orient_action(my_pos, station_pos)
        ori = state.to_dict()["players"][agent_idx]["orientation"]
        if desired is not None:
            want_vec = ACTION_DELTAS[desired]
            if tuple(ori) != tuple(want_vec):
                return desired
    return INTERACT


def sidestep_action(mdp, pos, other_pos):
    for a, (dx, dy) in ACTION_DELTAS.items():
        np = (pos[0] + dx, pos[1] + dy)
        if is_walkable(mdp, np) and np != other_pos:
            return a
    return STAY


def resolve_conflicts(mdp, state, a0: int, a1: int) -> Tuple[int, int]:
    """Avoid stepping into each other; yield on collision (try sidestep first)."""
    p0 = state.players[0].position
    p1 = state.players[1].position

    def cand(pos, a):
        if a in MOVE_ACTIONS:
            dx, dy = ACTION_DELTAS[a]
            return (pos[0] + dx, pos[1] + dy)
        return pos

    def valid_move(pos, a):
        return a in MOVE_ACTIONS and is_walkable(mdp, cand(pos, a))

    cand0 = cand(p0, a0)
    cand1 = cand(p1, a1)
    move0 = valid_move(p0, a0)
    move1 = valid_move(p1, a1)

    if move1 and cand1 == p0:
        a1 = sidestep_action(mdp, p1, p0)

    if move0 and cand0 == p1:
        alt1 = sidestep_action(mdp, p1, p0)
        if alt1 != STAY:
            a1 = alt1
        else:
            a0 = STAY

    cand0 = cand(p0, a0)
    cand1 = cand(p1, a1)
    move0 = valid_move(p0, a0)
    move1 = valid_move(p1, a1)

    same_target = move0 and move1 and (cand0 == cand1)
    swap = move0 and move1 and (cand0 == p1 and cand1 == p0)
    if same_target or swap:
        alt1 = sidestep_action(mdp, p1, p0)
        a1 = alt1 if alt1 != STAY else STAY

    return a0, a1


# ============================================================
# Explanation logic (AI teammate)
# ============================================================
KEY_EVENTS = {
    "picked_onion",
    "placed_or_dropped_onion",
    "picked_dish",
    "placed_or_dropped_dish",
    "picked_soup",
    "delivered_or_dropped_soup",
}


def goal_server(pot_onions_est: int, cook_kick_steps: int, held1: Optional[str]) -> str:
    if cook_kick_steps > 0:
        return "kickstart cooking"
    if held1 is not None and "soup" in str(held1).lower():
        return "deliver soup to serving"
    if held1 == "dish":
        if pot_onions_est >= 3:
            return "pick up soup from pot"
        return "wait for pot to fill"
    if held1 is None:
        return "collect dish"
    return "recover / wait"


def format_explanation(agent_name: str, goal: str, action_idx: int, mode: str) -> str:
    a_name = ACTION_NAMES.get(action_idx, str(action_idx))
    if mode == "action":
        return f"{agent_name}: action={a_name}"
    if mode == "goal":
        return f"{agent_name}: {goal}"
    return f"{agent_name}: {goal} | action={a_name}"


def instruction_lines(condition: str) -> List[str]:
    return [
        "You are BLUE (0). AI teammate is RED (1).",
        "Objective: make and serve onion soup as fast as possible.",
        "How to make soup:",
        "1) Go to O (onion) and press SPACE to pick up an onion.",
        "2) Go next to P (pot) and press SPACE to add the onion.",
        "3) Repeat until the pot shows 3/3 onions.",
        "4) Wait briefly for cooking, then the AI will pick up soup using a dish.",
        "5) The AI takes soup to S (serve) to score points.",
        "Controls: WASD/Arrows move, SPACE interact, P pause, R restart, Q/ESC quit.",
        f"Condition: {condition.upper()}  (Explanations {'ON' if condition=='exp' else 'OFF'})",
    ]


# ============================================================
# Renderer (your nice themed version)
# ============================================================
@dataclass
class RenderConfig:
    tile_size: int = 70
    margin: int = 2
    hud_height: int = 300
    min_width: int = 1000


class PygameRenderer:
    PALETTE = {
        "bg": (12, 12, 14),
        "floor_a": (32, 33, 38),
        "floor_b": (28, 29, 34),
        "counter": (78, 78, 86),
        "counter_edge": (105, 105, 120),
        "grid_edge": (18, 18, 22),
        "station_O": (230, 200, 60),
        "station_D": (110, 200, 230),
        "station_P": (190, 120, 230),
        "station_S": (120, 220, 140),
        "station_T": (230, 110, 110),
        "text": (235, 235, 240),
        "player0": (80, 160, 255),
        "player1": (255, 120, 120),
        "outline": (10, 10, 10),
        "hud_bg": (10, 10, 12, 210),
        "banner_bg": (22, 22, 28, 235),
        "banner_edge": (255, 255, 255, 30),
    }

    def __init__(self, mdp, cfg: RenderConfig):
        self.mdp = mdp
        self.cfg = cfg
        self.rows = len(mdp.terrain_mtx)
        self.cols = len(mdp.terrain_mtx[0])

        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont("consolas", 18)
        self.big = pygame.font.SysFont("consolas", 22, bold=True)
        self.banner = pygame.font.SysFont("consolas", 20, bold=True)

        self.grid_w = self.cols * cfg.tile_size
        self.grid_h = self.rows * cfg.tile_size

        w = max(self.grid_w, cfg.min_width)
        h = self.grid_h + cfg.hud_height

        self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        pygame.display.set_caption("Overcooked Human Study Runner")

    def _grid_offset_x(self):
        w = self.screen.get_width()
        return max(0, (w - self.grid_w) // 2)

    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        words = text.split(" ")
        lines = []
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            if self.font.size(test)[0] <= max_width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    @staticmethod
    def _obj_badge(name: str):
        s = str(name).lower()
        if "onion" in s:
            return "O", (200, 200, 0)
        if "dish" in s:
            return "D", (230, 230, 230)
        if "soup" in s:
            return "S", (255, 170, 0)
        return "?", (180, 180, 180)

    def _tile_color(self, ch: str, x: int, y: int):
        if ch == " ":
            return self.PALETTE["floor_a"] if (x + y) % 2 == 0 else self.PALETTE["floor_b"]
        if ch == "X":
            return self.PALETTE["counter"]
        if ch == "O":
            return self.PALETTE["station_O"]
        if ch == "D":
            return self.PALETTE["station_D"]
        if ch == "P":
            return self.PALETTE["station_P"]
        if ch == "S":
            return self.PALETTE["station_S"]
        if ch == "T":
            return self.PALETTE["station_T"]
        return self.PALETTE["floor_a"]

    def _draw_outlined_rect(self, rect, fill, edge, edge_w=2, radius=10):
        pygame.draw.rect(self.screen, fill, rect, border_radius=radius)
        pygame.draw.rect(self.screen, edge, rect, width=edge_w, border_radius=radius)

    def _draw_outlined_circle(self, center, r, fill, edge, edge_w=3):
        pygame.draw.circle(self.screen, fill, center, r)
        pygame.draw.circle(self.screen, edge, center, r, width=edge_w)

    def _draw_progress_bar(self, x, y, w, h, frac, back=(0, 0, 0), fore=(255, 255, 255)):
        frac = max(0.0, min(1.0, frac))
        pygame.draw.rect(self.screen, back, pygame.Rect(x, y, w, h), border_radius=6)
        pygame.draw.rect(self.screen, fore, pygame.Rect(x, y, int(w * frac), h), border_radius=6)

    def draw(self, state, hud_lines: List[str], pot_onions_est: int, banner_text: str = ""):
        cfg = self.cfg
        ts = cfg.tile_size
        x0 = self._grid_offset_x()

        self.screen.fill(self.PALETTE["bg"])

        for y in range(self.rows):
            for x in range(self.cols):
                ch = self.mdp.terrain_mtx[y][x]
                rect = pygame.Rect(
                    x0 + x * ts + cfg.margin,
                    y * ts + cfg.margin,
                    ts - 2 * cfg.margin,
                    ts - 2 * cfg.margin,
                )

                fill = self._tile_color(ch, x, y)

                if ch == "X":
                    self._draw_outlined_rect(rect, fill, self.PALETTE["counter_edge"], edge_w=2, radius=10)
                elif ch in ["O", "D", "P", "S", "T"]:
                    self._draw_outlined_rect(rect, fill, self.PALETTE["grid_edge"], edge_w=2, radius=12)
                else:
                    pygame.draw.rect(self.screen, fill, rect, border_radius=10)
                    pygame.draw.rect(self.screen, self.PALETTE["grid_edge"], rect, width=1, border_radius=10)

                if ch in ["O", "D", "P", "S", "T"]:
                    label = self.big.render(ch, True, (15, 15, 18))
                    self.screen.blit(label, (x0 + x * ts + ts // 2 - 8, y * ts + ts // 2 - 12))

                if ch == "P":
                    bar_w = rect.w - 16
                    bar_h = 10
                    bx = rect.x + 8
                    by = rect.y + rect.h - 18
                    frac = max(0.0, min(1.0, pot_onions_est / 3.0))
                    self._draw_progress_bar(bx, by, bar_w, bar_h, frac, back=(0, 0, 0), fore=(250, 250, 250))

        objs = state.to_dict().get("objects", [])
        for o in objs:
            name = o.get("name", "")
            (ox, oy) = o.get("position", (0, 0))
            cx = x0 + ox * ts + ts // 2
            cy = oy * ts + ts // 2

            badge, col = self._obj_badge(name)
            self._draw_outlined_circle((cx, cy), 12, col, self.PALETTE["outline"], edge_w=3)
            txt = self.font.render(badge, True, (0, 0, 0))
            self.screen.blit(txt, (cx - 6, cy - 10))

        p0 = state.players[0].position
        p1 = state.players[1].position
        for idx, (px, py) in enumerate([p0, p1]):
            cx = x0 + px * ts + ts // 2
            cy = py * ts + ts // 2
            fill = self.PALETTE["player0"] if idx == 0 else self.PALETTE["player1"]
            self._draw_outlined_circle((cx, cy), 16, fill, self.PALETTE["outline"], edge_w=4)
            lab = self.big.render(str(idx), True, (0, 0, 0))
            self.screen.blit(lab, (cx - 6, cy - 12))

        for idx, pos in enumerate([p0, p1]):
            player = state.players[idx]
            obj = getattr(player, "held_object", None)
            if obj is None:
                continue
            nm = getattr(obj, "name", None) or getattr(obj, "obj_type", None) or str(obj)
            badge, col = self._obj_badge(nm)

            px, py = pos
            cx = x0 + px * ts + ts // 2
            cy = py * ts + ts // 2 - 26

            self._draw_outlined_circle((cx, cy), 12, col, self.PALETTE["outline"], edge_w=3)
            txt = self.font.render(badge, True, (0, 0, 0))
            self.screen.blit(txt, (cx - 6, cy - 10))

        hud_y = self.grid_h
        w = self.screen.get_width()

        hud_surf = pygame.Surface((w, cfg.hud_height), pygame.SRCALPHA)
        hud_surf.fill(self.PALETTE["hud_bg"])
        self.screen.blit(hud_surf, (0, hud_y))

        y = hud_y + 10
        max_w = w - 20

        if banner_text:
            banner_h = 38
            banner_surf = pygame.Surface((w, banner_h), pygame.SRCALPHA)
            banner_surf.fill(self.PALETTE["banner_bg"])
            self.screen.blit(banner_surf, (0, hud_y))

            line = pygame.Surface((w, 2), pygame.SRCALPHA)
            line.fill(self.PALETTE["banner_edge"])
            self.screen.blit(line, (0, hud_y + banner_h - 2))

            lines = self._wrap_text(banner_text, max_w)
            text = self.banner.render(lines[0], True, self.PALETTE["text"])
            self.screen.blit(text, (10, hud_y + 8))
            y = hud_y + banner_h + 10

        for line_txt in hud_lines:
            for wrapped in self._wrap_text(line_txt, max_w):
                txt = self.font.render(wrapped, True, self.PALETTE["text"])
                self.screen.blit(txt, (10, y))
                y += 22
                if y > hud_y + cfg.hud_height - 22:
                    break

        pygame.display.flip()


# ============================================================
# Input (pause/reset)
# ============================================================
def read_human_action() -> Tuple[Optional[int], bool, Optional[str]]:
    quit_flag = False
    command = None

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_flag = True
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                quit_flag = True
            elif event.key == pygame.K_p:
                command = "pause"
            elif event.key == pygame.K_r:
                command = "reset"

    if quit_flag:
        return None, True, None

    if command is not None:
        return STAY, False, command

    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        return INTERACT, False, None

    if keys[pygame.K_w] or keys[pygame.K_UP]:
        return 0, False, None
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        return 1, False, None
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        return 2, False, None
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        return 3, False, None

    return STAY, False, None


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    mdp = OvercookedGridworld.from_layout_name(args.layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=args.horizon)
    env.reset()

    start_state = env.state
    p0_start = start_state.players[0].position
    p1_start = start_state.players[1].position

    onion_tiles = find_tiles(mdp, "O")
    dish_tiles = find_tiles(mdp, "D")
    pot_tiles = find_tiles(mdp, "P")
    serve_tiles = find_tiles(mdp, "S")
    counter_tiles = find_tiles(mdp, "X")

    onion_tile, onion_adj = choose_station_tile(mdp, onion_tiles, prefer_near=p0_start)
    dish_tile, dish_adj = choose_station_tile(mdp, dish_tiles, prefer_near=p1_start)
    pot_tile, pot_adj = choose_station_tile(mdp, pot_tiles, prefer_near=p0_start)
    serve_tile, serve_adj = choose_station_tile(mdp, serve_tiles, prefer_near=p1_start)
    server_wait = serve_adj[0] if serve_adj else p1_start

    renderer = PygameRenderer(mdp, RenderConfig())

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cond = args.condition
    run_id = (args.run_id or "").strip()
    run_tag = f"_{run_id}" if run_id else ""

    csv_path = os.path.join(args.out_dir, f"human_{args.participant}_{cond}_{args.layout}{run_tag}_{ts}.csv")
    summary_path = os.path.join(args.out_dir, f"human_{args.participant}_{cond}_{args.layout}{run_tag}_{ts}_summary.json")

    EXPLANATIONS_ON = (args.condition == "exp")

    total_game_reward = 0.0
    total_shaped_reward = 0.0
    explanation_count = 0
    prev_goal_ai: Optional[str] = None
    last_exp_ai: str = ""
    exp_flash_until = 0.0

    paused = False
    deliver_flash_until = 0.0
    last_delivery_reward = 0.0

    pot_onions_est = 0
    cook_kick_steps = 0

    # Dish stash memory
    stash_counter_tile: Optional[Tuple[int, int]] = None
    stash_counter_adj: Optional[List[Tuple[int, int]]] = None
    pending_stash_target: Optional[Tuple[int, int]] = None
    pending_stash_adj: Optional[List[Tuple[int, int]]] = None

    episode = 0
    t_total = 0
    t_ep = 0

    start_time = time.time()
    instr_until = time.time() + INSTR_SEC
    clock = pygame.time.Clock()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "participant", "run_id", "condition", "layout",
            "episode", "t_total", "t_ep",
            "a_human", "a_ai",
            "p0", "p1", "p0_next", "p1_next",
            "held0_before", "held1_before", "held0_after", "held1_after",
            "event0", "event1",
            "r_game", "r_shaped_total",
            "total_game_reward", "total_shaped_reward",
            "pot_onions_est", "cook_kick_steps",
            "goal_ai", "exp_ai"
        ])

        done = False
        quit_flag = False

        while not done and not quit_flag:
            state = env.state
            a0, quit_flag, command = read_human_action()
            if quit_flag:
                break
            assert a0 is not None

            if command == "pause":
                paused = not paused

            if command == "reset":
                env.reset()
                episode += 1
                t_ep = 0

                total_game_reward = 0.0
                total_shaped_reward = 0.0
                explanation_count = 0
                prev_goal_ai = None
                last_exp_ai = ""
                exp_flash_until = 0.0

                deliver_flash_until = 0.0
                last_delivery_reward = 0.0

                pot_onions_est = 0
                cook_kick_steps = 0

                stash_counter_tile = None
                stash_counter_adj = None
                pending_stash_target = None
                pending_stash_adj = None

                paused = False
                start_time = time.time()
                instr_until = time.time() + INSTR_SEC
                continue

            now = time.time()

            if paused:
                elapsed = now - start_time
                hud = [
                    f"Participant: {args.participant} | Run: {run_id or '-'} | Condition: {cond.upper()} | Layout: {args.layout}",
                    f"Horizon: {args.horizon} steps | FPS: {args.fps}",
                    "Controls: WASD/Arrows move | SPACE interact | P pause | R restart | Q/ESC quit",
                    f"PAUSED | score={total_game_reward:.1f} shaped={total_shaped_reward:.1f} time={elapsed:.1f}s",
                    f"Pot: {pot_onions_est}/3 | kick_steps={cook_kick_steps}",
                ]
                banner = "PAUSED (press P to resume)"
                if now < instr_until:
                    hud = instruction_lines(cond) + [""] + hud
                    banner = "INSTRUCTIONS: " + instruction_lines(cond)[0]
                renderer.draw(env.state, hud, pot_onions_est, banner_text=banner)
                clock.tick(args.fps)
                continue

            # --- pre-step state ---
            held0_b = held_name(state.players[0])
            held1_b = held_name(state.players[1])
            p0 = state.players[0].position
            p1 = state.players[1].position
            pot_full_est = (pot_onions_est >= 3)

            # Make SPACE forgiving for human (fix)
            a0 = human_space_autoorient(state, p0, held0_b, a0, onion_tile, pot_tile)

            # clear pending stash each step
            pending_stash_target = None
            pending_stash_adj = None

            # ---------------------------------------------------------
            # AI policy
            # ---------------------------------------------------------
            if cook_kick_steps > 0:
                if held1_b == "dish":
                    objs_now = state_objects_dict(state)
                    if stash_counter_tile is not None and stash_counter_tile not in objs_now and stash_counter_adj:
                        target_ct, target_adj = stash_counter_tile, stash_counter_adj
                    else:
                        target_ct, target_adj = choose_empty_counter(mdp, state, counter_tiles, prefer_near=p1)

                    if target_ct is not None:
                        pending_stash_target = target_ct
                        pending_stash_adj = target_adj
                        a1 = go_and_interact(mdp, state, 1, target_ct, target_adj)
                    else:
                        a1 = go_and_interact(mdp, state, 1, pot_tile, pot_adj)
                else:
                    a1 = go_and_interact(mdp, state, 1, pot_tile, pot_adj)

            else:
                if held1_b is None and stash_counter_tile is not None and counter_has_dish(state, stash_counter_tile) and stash_counter_adj:
                    a1 = go_and_interact(mdp, state, 1, stash_counter_tile, stash_counter_adj)
                elif pot_full_est and held1_b is None:
                    a1 = go_and_interact(mdp, state, 1, dish_tile, dish_adj)
                else:
                    if held1_b is not None and "soup" in str(held1_b).lower():
                        a1 = go_and_interact(mdp, state, 1, serve_tile, serve_adj)
                    elif held1_b == "dish":
                        if pot_full_est:
                            a1 = go_and_interact(mdp, state, 1, pot_tile, pot_adj)
                        else:
                            if p1 != server_wait:
                                a1 = bfs_first_step(mdp, p1, {server_wait})
                            else:
                                a1 = STAY
                    elif held1_b is None:
                        a1 = go_and_interact(mdp, state, 1, dish_tile, dish_adj)
                    else:
                        if p1 != server_wait:
                            a1 = bfs_first_step(mdp, p1, {server_wait})
                        else:
                            a1 = STAY

            # Resolve conflicts
            a0, a1 = resolve_conflicts(mdp, state, a0, a1)

            # Step
            joint_action = (Action.INDEX_TO_ACTION[a0], Action.INDEX_TO_ACTION[a1])
            _next_state, r_game, done, info = env.step(joint_action)

            if float(r_game) > 0:
                last_delivery_reward = float(r_game)
                deliver_flash_until = time.time() + 1.5

            shaped = sum(info.get("shaped_r_by_agent", []))
            total_game_reward += float(r_game)
            total_shaped_reward += float(shaped)

            # --- post-step state ---
            state2 = env.state
            p0n = state2.players[0].position
            p1n = state2.players[1].position
            held0_a = held_name(state2.players[0])
            held1_a = held_name(state2.players[1])

            event0 = infer_event(held0_b, held0_a)
            event1 = infer_event(held1_b, held1_a)

            # Record stash location on successful drop
            if pending_stash_target is not None:
                dish_dropped = (held1_b == "dish" and held1_a is None and a1 == INTERACT and (p1 in (pending_stash_adj or [])))
                if dish_dropped:
                    stash_counter_tile = pending_stash_target
                    stash_counter_adj = pending_stash_adj

            # ---------------------------------------------------------
            # Pot tracking (your heuristic, but WITHOUT bogus "rejected means full")
            # ---------------------------------------------------------
            chef_at_pot = (p0 in pot_adj)
            old_pot = pot_onions_est

            pot_add_success = int(chef_at_pot and a0 == INTERACT and held0_b == "onion" and held0_a is None)
            if pot_add_success:
                pot_onions_est = min(3, pot_onions_est + 1)

            # Start kick when pot becomes full
            if old_pot < 3 and pot_onions_est == 3:
                cook_kick_steps = KICK_STEPS

            # Reset once AI picks soup
            if event1 == "picked_soup" or (held1_a is not None and "soup" in str(held1_a).lower()):
                pot_onions_est = 0
                cook_kick_steps = 0

            # Decrement kick steps only when AI actually kicks empty-handed at pot
            ai_at_pot = (p1 in pot_adj)
            ai_kick_done = int(ai_at_pot and a1 == INTERACT and held1_b is None)
            if cook_kick_steps > 0 and ai_kick_done:
                cook_kick_steps -= 1

            # Explanations
            goal_ai = goal_server(pot_onions_est, cook_kick_steps, held1_b)
            exp_ai = ""

            if EXPLANATIONS_ON:
                goal_changed = (prev_goal_ai is None) or (goal_ai != prev_goal_ai)
                trig = True
                if args.goal_change_only:
                    trig = goal_changed
                if args.event_trigger:
                    if (event1 in KEY_EVENTS) or (event0 in KEY_EVENTS):
                        trig = True

                if trig:
                    exp_ai = format_explanation("AI", goal_ai, a1, args.explain_mode)
                    last_exp_ai = exp_ai
                    explanation_count += 1
                    exp_flash_until = time.time() + 2.0

                prev_goal_ai = goal_ai

            now = time.time()

            # Banner priority: instructions > delivery > explanation
            banner_text = ""
            hud_prefix: List[str] = []

            if now < instr_until:
                banner_text = "INSTRUCTIONS: " + instruction_lines(cond)[0]
                hud_prefix = instruction_lines(cond) + [""]

            elif now < deliver_flash_until:
                banner_text = f"SOUP DELIVERED! +{last_delivery_reward:.0f}"

            elif EXPLANATIONS_ON and last_exp_ai and now < exp_flash_until:
                banner_text = f"EXPLANATION: {last_exp_ai}"

            elapsed = now - start_time
            hud = [
                f"Participant: {args.participant} | Run: {run_id or '-'} | Condition: {cond.upper()} | Layout: {args.layout}",
                f"Horizon: {args.horizon} steps | FPS: {args.fps}",
                "Controls: WASD/Arrows move | SPACE interact | P pause | R restart | Q/ESC quit",
                f"episode={episode} t_ep={t_ep}/{args.horizon} score={total_game_reward:.1f} shaped={total_shaped_reward:.1f} time={elapsed:.1f}s",
                f"Pot: {pot_onions_est}/3 | kick_steps={cook_kick_steps}",
                f"AI goal: {goal_ai if EXPLANATIONS_ON else '(hidden)'}",
                f"AI last explanation: {last_exp_ai if EXPLANATIONS_ON else '(hidden)'}",
                "Legend: Blue=You (0), Red=AI (1). Held items shown above heads.",
            ]
            hud = hud_prefix + hud

            renderer.draw(state2, hud, pot_onions_est, banner_text=banner_text)

            w.writerow([
                args.participant, run_id, cond, args.layout,
                episode, t_total, t_ep,
                a0, a1,
                p0, p1, p0n, p1n,
                held0_b, held1_b, held0_a, held1_a,
                event0, event1,
                float(r_game), float(shaped),
                float(total_game_reward), float(total_shaped_reward),
                int(pot_onions_est), int(cook_kick_steps),
                goal_ai if EXPLANATIONS_ON else "",
                exp_ai if EXPLANATIONS_ON else ""
            ])

            t_total += 1
            t_ep += 1
            clock.tick(args.fps)

    duration = time.time() - start_time
    summary = {
        "participant": args.participant,
        "run_id": run_id,
        "condition": cond,
        "layout": args.layout,
        "horizon": args.horizon,
        "episodes_played": episode + 1,
        "steps_total": t_total,
        "duration_sec": duration,
        "total_game_reward": total_game_reward,
        "total_shaped_reward": total_shaped_reward,
        "explanations_emitted": explanation_count,
        "csv_path": csv_path,
    }
    with open(summary_path, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)

    pygame.quit()
    print("Saved log:", csv_path)
    print("Saved summary:", summary_path)
    print("Summary:", summary)


if __name__ == "__main__":
    main()