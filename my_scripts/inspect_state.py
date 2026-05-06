from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked

mdp = OvercookedGridworld.from_layout_name("cramped_room")
base_env = OvercookedEnv.from_mdp(mdp, horizon=20)
env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

def safe_player_info(state, i):
    p = state.players[i]
    pos = getattr(p, "position", None)
    held = getattr(p, "held_object", None)
    held_name = None
    if held is not None:
        held_name = getattr(held, "name", None) or getattr(held, "obj_type", None) or str(held)
    return pos, held_name

env.reset()

for t in range(5):
    state = env.base_env.state
    p0, h0 = safe_player_info(state, 0)
    p1, h1 = safe_player_info(state, 1)
    print(f"\nBefore step {t}: p0={p0}, held0={h0} | p1={p1}, held1={h1}")

    a0 = env.action_space.sample()
    a1 = env.action_space.sample()
    obs, r, done, info = env.step((a0, a1))

    state2 = env.base_env.state
    p0b, h0b = safe_player_info(state2, 0)
    p1b, h1b = safe_player_info(state2, 1)
    print(f" After step {t}: p0={p0b}, held0={h0b} | p1={p1b}, held1={h1b} | r={r} shaped={info.get('shaped_r_by_agent')}")
