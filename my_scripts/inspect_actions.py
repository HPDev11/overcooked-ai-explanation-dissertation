from overcooked_ai_py.mdp.actions import Action

print("ALL_ACTIONS:", Action.ALL_ACTIONS)
# If this exists in your version, it helps mapping
if hasattr(Action, "INDEX_TO_ACTION"):
    print("INDEX_TO_ACTION:", Action.INDEX_TO_ACTION)
if hasattr(Action, "ACTION_TO_INDEX"):
    print("ACTION_TO_INDEX:", Action.ACTION_TO_INDEX)
