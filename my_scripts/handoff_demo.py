from collections import deque
from pprint import pprint

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked

# Actions: 0 up, 1 down, 2 right, 3 left, 4 stay, 5 interact
ACTION_DELTAS = {0:(0,-1), 1:(0,1), 2:(1,0), 3:(-1,0)}
STAY = 4
INTERACT = 5

LAYOUT = "cramped_room"
HORIZON = 200

mdp = OvercookedGridworld.from_layout_name(LAYOUT)
base_env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON)
env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

WALKABLE = {" "}  # from your terrain inspection

def in_bounds(pos):
    x, y = pos
    return 0 <= y < len(mdp.terrain_mtx) and 0 <= x < len(mdp.terrain_mtx[0])

def is_walkable(pos):
    if not in_bounds(pos):
        return False
    x, y = pos
    return mdp.terrain_mtx[y][x] in WALKABLE

def find_tiles(ch):
    tiles = []
    for y, row in enumerate(mdp.terrain_mtx):
        for x, c in enumerate(row):
            if c == ch:
                tiles.append((x, y))
    return tiles

def neighbors(pos):
    x, y = pos
    for a, (dx, dy) in ACTION_DELTAS.items():
        np = (x + dx, y + dy)
        if is_walkable(np):
            yield a, np

def bfs_path(start, goal):
    """Return list of actions to move from start to goal on walkable tiles."""
    if start == goal:
        return []
    q = deque([start])
    parent = {start: None}
    parent_action = {start: None}

    while q:
        cur = q.popleft()
        for a, nxt in neighbors(cur):
            if nxt not in parent:
                parent[nxt] = cur
                parent_action[nxt] = a
                if nxt == goal:
                    q.clear()
                    break
                q.append(nxt)

    if goal not in parent:
        raise RuntimeError(f"No path from {start} to {goal}")

    # reconstruct
    actions = []
    cur = goal
    while parent[cur] is not None:
        actions.append(parent_action[cur])
        cur = parent[cur]
    actions.reverse()
    return actions

def step_joint(a0, a1):
    obs, r, done, info = env.step((a0, a1))
    return obs, r, done, info

def held_name(player):
    obj = getattr(player, "held_object", None)
    if obj is None:
        return None
    return getattr(obj, "name", None) or getattr(obj, "obj_type", None) or str(obj)

def orient_toward(from_pos, target_pos):
    """Return a MOVE action that points from from_pos toward adjacent target_pos."""
    fx, fy = from_pos
    tx, ty = target_pos
    dx, dy = tx - fx, ty - fy
    for a, (adx, ady) in ACTION_DELTAS.items():
        if (adx, ady) == (dx, dy):
            return a
    raise RuntimeError("Target not adjacent; cannot orient")

def has_adjacent_floor(target_pos):
    x, y = target_pos
    for dx, dy in [(0,-1),(0,1),(1,0),(-1,0)]:
        p = (x+dx, y+dy)
        if is_walkable(p):
            return True
    return False

def pick_adjacent_floor(target_pos):
    """Pick a walkable tile adjacent to target_pos."""
    x, y = target_pos
    for dx, dy in [(0,-1),(0,1),(1,0),(-1,0)]:
        p = (x+dx, y+dy)
        if is_walkable(p):
            return p
    raise RuntimeError(f"No adjacent walkable tile for target {target_pos}")

# --- Start ---
env.reset()

# Find an onion dispenser tile and a counter tile
onions = [t for t in find_tiles("O") if has_adjacent_floor(t)]
if not onions:
    raise RuntimeError("No onion dispenser tile with adjacent floor found")
onion_tile = onions[0]
onion_adj = pick_adjacent_floor(onion_tile)

state = env.base_env.state
p1_start = state.players[1].position

candidates = []
for t in find_tiles("X"):
    if not has_adjacent_floor(t):
        continue
    try:
        adj = pick_adjacent_floor(t)
    except RuntimeError:
        continue
    # Avoid using the same interaction tile as onion pickup, and avoid where agent1 starts
    if adj == onion_adj or adj == p1_start:
        continue
    candidates.append((t, adj))

if not candidates:
    raise RuntimeError("No suitable counter found (try relaxing filters)")

counter_tile, counter_adj = candidates[0]


print("Onion tile:", onion_tile, "adj floor:", onion_adj)
print("Counter tile:", counter_tile, "adj floor:", counter_adj)

# Helper: move one agent along a path while the other stays
def move_agent(agent_idx, actions):
    for a in actions:
        if agent_idx == 0:
            step_joint(a, STAY)
        else:
            step_joint(STAY, a)

# 1) Move agent 0 to onion_adj
state = env.base_env.state
p0 = state.players[0].position
path0 = bfs_path(p0, onion_adj)
move_agent(0, path0)

# 2) Orient agent 0 toward onion tile (attempt move into it) then INTERACT to pick onion
state = env.base_env.state
p0 = state.players[0].position
a_face_onion = orient_toward(p0, onion_tile)
# attempt move into dispenser (not walkable) -> sets orientation
step_joint(a_face_onion, STAY)
# interact to pick
step_joint(INTERACT, STAY)

state = env.base_env.state
print("After pickup: held0 =", held_name(state.players[0]))

# 3) Move agent 0 to counter_adj
state = env.base_env.state
p0 = state.players[0].position
path0b = bfs_path(p0, counter_adj)
move_agent(0, path0b)

# 4) Orient toward counter then INTERACT to place on counter
state = env.base_env.state
p0 = state.players[0].position
a_face_counter = orient_toward(p0, counter_tile)
step_joint(a_face_counter, STAY)   # attempt move into counter (blocked) sets orientation
step_joint(INTERACT, STAY)         # place onion on counter

state = env.base_env.state
print("After place: held0 =", held_name(state.players[0]))
print("Objects now (should include onion on counter):")
pprint(state.to_dict()["objects"])
# Move agent0 off counter_adj so agent1 can stand there
state = env.base_env.state
p1 = state.players[1].position

moved_off = False
for a, np in neighbors(counter_adj):
    if np != counter_adj and np != p1:
        step_joint(a, STAY)  # agent0 moves, agent1 stays
        moved_off = True
        break

if not moved_off:
    raise RuntimeError("Could not vacate counter_adj (no free neighboring floor)")
# IMPORTANT: move agent 0 away so agent 1 can stand on counter_adj
# (For cramped_room, moving right from (1,1) -> (2,1) is walkable)
step_joint(2, STAY)  # agent0: right, agent1: stay

# 5) Move agent 1 to counter_adj
state = env.base_env.state
p1 = state.players[1].position
path1 = bfs_path(p1, counter_adj)
move_agent(1, path1)

# 6) Orient agent 1 toward counter then INTERACT to pick up
state = env.base_env.state
p1 = state.players[1].position
if p1 != counter_adj:
    raise RuntimeError(f"Agent 1 is at {p1}, expected {counter_adj} (not adjacent to counter)")
a1_face_counter = orient_toward(p1, counter_tile)
step_joint(STAY, a1_face_counter)
step_joint(STAY, INTERACT)

state = env.base_env.state
print("After pickup by agent 1: held1 =", held_name(state.players[1]))
print("Objects now (should be empty or reduced):")
pprint(state.to_dict()["objects"])

print("\nHandoff demo complete.")
