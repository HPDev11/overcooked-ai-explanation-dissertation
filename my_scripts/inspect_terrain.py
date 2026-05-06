from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

mdp = OvercookedGridworld.from_layout_name("cramped_room")
tm = mdp.terrain_mtx

chars = set()
for row in tm:
    for c in row:
        chars.add(c)

print("Unique terrain chars:", sorted(chars))
print("Example terrain (first 5 rows):")
for row in tm[:5]:
    print("".join(row))
