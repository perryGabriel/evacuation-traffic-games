from evacuation_vug.network import load_network


def test_network_parses_and_routes_unique():
    net = load_network("configs/networks/default_tree.yaml")
    assert len(net.generators) == 6
    for g in net.generators:
        route = net.route_to_safe(g)
        assert route[0] == g
        assert route[-1] == net.safe_node
