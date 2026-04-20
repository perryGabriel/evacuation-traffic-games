from evacuation_vug.network import load_network
from evacuation_vug.surrogate import welfare


def test_diminishing_returns_toy_tokens():
    net = load_network("configs/networks/toy_tree.yaml")
    cfg = {
        "alpha": {"g1": 1.0, "g2": 1.0},
        "phi_type": {"g1": "geometric", "g2": "geometric"},
        "phi_params": {"g1": {"beta": 0.5}, "g2": {"beta": 0.5}},
    }
    a = {("i1", "g1->i1", 0)}
    b = a | {("i1", "g1->i1", 1)}
    x = ("i1", "g1->i1", 2)
    wa = welfare(net, a, cfg["alpha"], cfg["phi_type"], cfg["phi_params"])
    wax = welfare(net, a | {x}, cfg["alpha"], cfg["phi_type"], cfg["phi_params"])
    wb = welfare(net, b, cfg["alpha"], cfg["phi_type"], cfg["phi_params"])
    wbx = welfare(net, b | {x}, cfg["alpha"], cfg["phi_type"], cfg["phi_params"])
    assert (wax - wa) >= (wbx - wb)
