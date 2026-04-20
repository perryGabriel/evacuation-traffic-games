from evacuation_vug.equilibria import enumerate_pure_nash
from evacuation_vug.network import load_network


def test_exact_enumeration_and_poa_bounds():
    net = load_network("configs/networks/toy_tree.yaml")
    cfg = {
        "alpha": {"g1": 1.0, "g2": 1.0},
        "phi_type": {"g1": "capped", "g2": "capped"},
        "phi_params": {"g1": {"D": 2}, "g2": {"D": 2}},
    }
    out = enumerate_pure_nash(net, horizon=2, surrogate_cfg=cfg)
    assert out.optimal_welfare >= out.best_equilibrium_welfare
    assert 0.0 <= out.poa <= 1.0
