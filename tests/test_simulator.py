from evacuation_vug.network import load_network
from evacuation_vug.policies import build_policy
from evacuation_vug.simulator import Simulator


def test_simulator_runs_and_updates_metadata():
    net = load_network("configs/networks/toy_tree.yaml")
    sim = Simulator(net, generation_rates={"g1": 0.8, "g2": 0.8}, seed=1)
    result = sim.run(horizon=20, policy=build_policy("all_open"))
    assert result.horizon == 20
    assert all(v.total_age >= 0 for v in result.completed + result.unfinished)
    assert len(result.schedule) == 20
