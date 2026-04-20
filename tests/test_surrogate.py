from evacuation_vug.network import load_network
from evacuation_vug.surrogate import welfare, utilities_by_intersection


def _cfg():
    return {
        "alpha": {"g1": 1.0, "g2": 1.0},
        "phi_type": {"g1": "capped", "g2": "capped"},
        "phi_params": {"g1": {"D": 2}, "g2": {"D": 2}},
    }


def test_welfare_empty_zero():
    net = load_network("configs/networks/toy_tree.yaml")
    cfg = _cfg()
    assert welfare(net, set(), cfg["alpha"], cfg["phi_type"], cfg["phi_params"]) == 0.0


def test_monotonicity_and_marginal_utility():
    net = load_network("configs/networks/toy_tree.yaml")
    cfg = _cfg()
    s1 = {("i1", "g1->i1", 0)}
    s2 = s1 | {("i1", "g2->i1", 1)}
    w1 = welfare(net, s1, cfg["alpha"], cfg["phi_type"], cfg["phi_params"])
    w2 = welfare(net, s2, cfg["alpha"], cfg["phi_type"], cfg["phi_params"])
    assert w2 >= w1
    utils = utilities_by_intersection(net, s2, cfg["alpha"], cfg["phi_type"], cfg["phi_params"])
    assert "i1" in utils and utils["i1"] == w2
