from evacuation_vug.policies import build_policy, PolicyContext


def _ctx():
    return PolicyContext(
        t=0,
        intersection_to_branches={"i1": ["a->i1", "b->i1"]},
        queue_waits={("i1", "a->i1"): 2, ("i1", "b->i1"): 3},
        queue_cumulative_waits={("i1", "a->i1"): 4, ("i1", "b->i1"): 1},
    )


def test_policy_actions_are_feasible():
    ctx = _ctx()
    for name in ["all_open", "cycle", "longest_current_wait", "longest_cumulative_wait"]:
        action = build_policy(name).choose(ctx)
        assert action["i1"] in ctx.intersection_to_branches["i1"]
