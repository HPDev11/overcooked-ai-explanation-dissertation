from pprint import pprint
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked

mdp = OvercookedGridworld.from_layout_name("cramped_room")
base_env = OvercookedEnv.from_mdp(mdp, horizon=50)
env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

env.reset()
state = env.base_env.state

print("State type:", type(state))
print("State attrs (filtered):", [a for a in dir(state) if "obj" in a.lower() or "objects" in a.lower()])

# try common fields safely
for name in ["objects", "all_objects", "unowned_objects_by_type", "to_dict"]:
    if hasattr(state, name):
        print(f"\nFound state.{name}")
        val = getattr(state, name)
        try:
            pprint(val() if callable(val) else val)
        except Exception as e:
            print("Could not print:", e)
