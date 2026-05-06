from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked

mdp = OvercookedGridworld.from_layout_name("cramped_room")
base_env = OvercookedEnv.from_mdp(mdp, horizon=50)
env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

obs = env.reset()

for t in range(5):
    a0 = env.action_space.sample()
    a1 = env.action_space.sample()
    obs, r, done, info = env.step((a0, a1))

    print(f"\nStep {t}")
    print("reward:", r)
    print("info keys:", list(info.keys()))

    # common fields (print only if they exist)
    if "shaped_r_by_agent" in info:
        print("shaped_r_by_agent:", info["shaped_r_by_agent"])
    if "episode_stats" in info:
        print("episode_stats:", info["episode_stats"])
