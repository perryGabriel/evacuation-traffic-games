import numpy as np
import networkx as nx
import random
from collections import abc,deque,defaultdict
import matplotlib.pyplot as plt

# ======================================================================================================================
# a method to generate a random directed tree with a specified number of nodes
# ======================================================================================================================

def generate_random_DAG(num_nodes, seed=None):
    if seed is not None:
        np.random.seed(seed)
    G = nx.random_labeled_tree(num_nodes, seed=seed)
    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes())
    distances_to_node_0 = nx.shortest_path_length(G, target=0)
    for u, v in G.edges():
        if distances_to_node_0[u] < distances_to_node_0[v]:
            DG.add_edge(v, u)
        else:
            DG.add_edge(u, v)
    source_nodes = [n for n in DG.nodes() if DG.in_degree(n) == 0]
    nodes_to_remove = []
    for s in source_nodes:
        if s != 0 and DG.has_edge(s, 0):
            nodes_to_remove.append(s)
    relabel_map = {}
    for s in source_nodes:
        if s not in nodes_to_remove and s != 0:
            relabel_map[s] = -s
    Modified_DG = DG.copy()
    Modified_DG.remove_nodes_from(nodes_to_remove)
    DG = nx.relabel_nodes(Modified_DG, relabel_map, copy=True)
    return DG

# ======================================================================================================================
# a method to build a networkx graph to use as a road network
# ======================================================================================================================

def build_road_network(edges=None, num_nodes=None, pos=None, seed=None, verbose=0):
    if isinstance(edges, dict):
        edges = list(edges.items())
    if isinstance(edges, abc.Iterable):
        all_nodes = np.unique([node for edge in edges for node in edge])

        # Create a directed graph from the edges according to the source/sink conventions
        G = nx.DiGraph()
        G.add_nodes_from(all_nodes)
        all_nodes = list(G.nodes())

    if not num_nodes is None:
        G = generate_random_DAG(num_nodes, seed)
        all_nodes = list(G.nodes())
        edges = list(G.edges())

    # Define edges (queues) directed towards the root (0)
    # Connections from generators to intersections
    for edge in edges:
        G.add_edge(*edge)

    # Initialize lists for node categories
    generators = []
    safe_region = []
    intersections = []

    # Iterate through all nodes in the graph G
    for node in G.nodes():
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)

        if in_degree == 0:
            generators.append(node)
        elif out_degree == 0:
            safe_region.append(node)
        else:
            intersections.append(node)

    # Add node attributes for type for easier visualization/categorization
    node_types = {}
    for node in generators:
        node_types[node] = 'generator'
    for node in intersections:
        node_types[node] = 'n intersection'
    for node in safe_region:
        node_types[node] = 'v safe Region'

    nx.set_node_attributes(G, node_types, 'type')

    if verbose > 0:
        # --- Visualization --- #
        plt.figure(figsize=(8, 6))

        if pos is None:
            pos = nx.spring_layout(G, seed=42)

        # Draw nodes with different colors based on type
        node_colors = []
        labels = {}
        for node in G.nodes():
            node_type = G.nodes[node]['type']
            labels[node] = f"${node_type[0]}_{{{node}}}$"
            if node_type == 'generator':
                node_colors.append('skyblue')
            elif node_type == 'n intersection':
                node_colors.append('salmon')
            else: # Safe Region
                node_colors.append('lightgreen')

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.9)
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrows=True, arrowsize=20, width=2, edge_color='gray')
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')

        plt.title("Traffic Network Tree Graph (Directed towards Root)")
        plt.axis('off')
        plt.show()

    return G

# ======================================================================================================================
# Car class, id manager, method to increment time counter
# ======================================================================================================================

class Car:
    def __init__(self, car_id, generator_id, current_queue=None):
        self.car_id = car_id
        self.generator_id = generator_id
        self.current_queue = current_queue  # The edge (u, v) where the car is located
        self.wait_time = 0                  # Time steps waiting in the current queue
        self.total_travel_time = 0          # Total time the car has spent in the system

    def __repr__(self):
        return f"Car(ID={self.car_id}, Queue={self.current_queue}, WaitTime={self.wait_time})"

class GlobalCarIDManager:
    _next_id = 0

    @staticmethod
    def get_next_id():
        GlobalCarIDManager._next_id += 1
        return GlobalCarIDManager._next_id

def increment_all_car_wait_times(all_queues_map):
    """Increments the wait_time and total_travel_time for all cars in all queues within all_queues_map."""
    for queue_id, traffic_queue in all_queues_map.items():
        for car in traffic_queue.get_cars_in_queue():
            car.wait_time += 1
            car.total_travel_time += 1

# ======================================================================================================================
# an Edge Queue model
# ======================================================================================================================

class TrafficQueue:
    def __init__(self, queue_id, wait_time_func=lambda S : 2, capacity=5):
        self.queue_id = queue_id
        self._cars = deque()  # Use deque for efficient FIFO operations
        self.wait_time_func = wait_time_func
        self.capacity = capacity

    def add_car(self, car, override_capacity=False):
        if not override_capacity and len(self._cars) >= self.capacity:
          raise ValueError(f"Queue {self.queue_id} is full. Cannot add more cars.")
        # Ensure car's current_queue matches this queue's id
        car.current_queue = self.queue_id
        self._cars.append(car)

    def count_cars(self):
        return len(self._cars)

    def is_full(self):
        return len(self._cars) >= self.capacity

    def peek_top_car_wait_time(self):
        if not self.is_empty():
            return self._cars[0].wait_time
        return None # Indicate empty queue, or use None/raise error

    def is_empty(self):
        return len(self._cars) == 0

    def can_pop_next_car(self):
        if not self.is_empty():
            return self._cars[0].wait_time >= self.wait_time_func(self.count_cars())
        return False

    def pop_car(self):
        if self.can_pop_next_car():
            return self._cars.popleft()
        return None  # Or raise an error if preferred

    def get_cars_in_queue(self):
        return list(self._cars)

    def __repr__(self):
        return f"TrafficQueue($q_{{{self.queue_id[0]},{self.queue_id[1]}}},\n\t Cars={self.count_cars()},\n\t TopWaitTime={self.peek_top_car_wait_time() if not self.is_empty() else 'N/A'})"

# ======================================================================================================================
# Generator node class and global manager
# ======================================================================================================================

class Generator:
    def __init__(self, generator_node_id, traffic_queue, generation_rate=0.5):
        self.generator_node_id = generator_node_id
        self.traffic_queue = traffic_queue
        self.generation_rate = generation_rate # Probability of generating a car per time step

    def step(self):
        """Simulates one time step for the generator, potentially adding a car to its queue."""
        if random.random() < self.generation_rate:
            car_id = GlobalCarIDManager.get_next_id()
            new_car = Car(car_id, self.generator_node_id)
            self.traffic_queue.add_car(new_car, override_capacity=True)
            # print(f"Generator {self.generator_node_id}: Car {car_id} generated and added to queue {self.traffic_queue.queue_id}.")

    def __repr__(self):
        return f"Generator(g_{{{self.generator_node_id}}}, Rate={self.generation_rate})"


class GlobalGenerator:
    def __init__(self, incoming_queues_adj_list, generation_rates=0.5):
      self.incoming_queues_adj_list = incoming_queues_adj_list
      self.all_generators = []
      if not isinstance(generation_rates, abc.Mapping):
          default_gen_rate = generation_rates
          generation_rates = {}
          for destination_node, source_queues_map in incoming_queues_adj_list.items():
              generation_rates.update({src : default_gen_rate for src in source_queues_map.keys() if src < 0})
      for destination_node, source_queues_map in self.incoming_queues_adj_list.items():
          for source_node, traffic_queue_obj in source_queues_map.items():
              if source_node < 0:  # Check if the source node is a generator (negative index)
                  generator_instance = Generator(source_node, traffic_queue_obj, generation_rates[source_node])
                  self.all_generators.append(generator_instance)

    def run_all_generators(self):
        """Calls the step method on all instantiated Generator objects."""
        for generator in self.all_generators:
            generator.step()

    def __repr__(self):
      return "Instantiated Generator objects:" + str(self.all_generators)

# ======================================================================================================================
# Intersection node class and global manager
# ======================================================================================================================

class Intersection:
    def __init__(self, intersection_id, policy, all_queues_map, graph_G):
        self.intersection_id = intersection_id
        self.policy = policy
        self.all_queues_map = all_queues_map
        self.graph_G = graph_G
        self.cum_cost = 0

        # Identify all incoming edge tuples (e.g., (u, intersection_id))
        self.incoming_queues_ids = []
        for u in self.graph_G.predecessors(self.intersection_id):
            self.incoming_queues_ids.append((u, self.intersection_id))

        # Identify all outgoing edge tuples (e.g., (intersection_id, v))
        self.outgoing_queues_ids = []
        for v in self.graph_G.successors(self.intersection_id):
            self.outgoing_queues_ids.append((self.intersection_id, v))

        # Initialize current open incoming queues (list of queue IDs)
        self.current_state_incoming_queue_ids = []

        # Policy management
        self.policy_step_counter = 0

        # Assume only one outgoing queue for simplicity in this tree structure
        # The problem statement implies a tree structure where a node has at most one successor node, except for the safe node
        if self.outgoing_queues_ids:
            self.outgoing_queue_id = self.outgoing_queues_ids[0]
        else:
            self.outgoing_queue_id = None # For the root node (0) this might be empty

    def step(self, completed_cars_log):
        """Simulates car movement through the intersection for one time step."""
        self.policy(self)
        self.policy_step_counter += 1

        if self.outgoing_queue_id is None:
            # This intersection is a sink (like the safe node, though intersections usually have out-edges)
            # No cars can move out if there's no outgoing queue.
            # For this problem's context, intersections always have an outgoing queue to 0 or another intersection.
            # The safe region (node 0) is the only node with 0 out-degree.
            return

        outgoing_traffic_queue = self.all_queues_map[self.outgoing_queue_id]

        np.random.shuffle(self.current_state_incoming_queue_ids)
        max_queue_top_time = 0
        for incoming_queue_id in self.current_state_incoming_queue_ids:
            incoming_traffic_queue = self.all_queues_map[incoming_queue_id]
            if incoming_traffic_queue.peek_top_car_wait_time() is not None and max_queue_top_time < incoming_traffic_queue.peek_top_car_wait_time():
                max_queue_top_time = incoming_traffic_queue.peek_top_car_wait_time()

            # Move cars from the open incoming queue to the outgoing queue
            while incoming_traffic_queue.can_pop_next_car():
                if outgoing_traffic_queue.is_full():
                    break
                car = incoming_traffic_queue.pop_car()
                if car:
                    # Check if the car is moving to the safe node (node 0)
                    if self.outgoing_queue_id[1] == 0: # outgoing_queue_id is (current_node, destination_node)
                        completed_cars_log.append({'car_id': car.car_id, 'origin': car.generator_id, 'total_travel_time': car.total_travel_time})
                        # Car exits simulation, no need to add to outgoing queue
                    else:
                        car.wait_time = 0  # Reset wait time upon moving to a new queue
                        outgoing_traffic_queue.add_car(car)

        self.cum_cost += max_queue_top_time

    def get_cost(self):
        return self.cum_cost / self.policy_step_counter

    def __repr__(self):
        return (
            f"Intersection(ID={self.intersection_id}, "
            f"Incoming Queues={[qid for qid in self.incoming_queues_ids]}, "
            f"Outgoing Queue={self.outgoing_queue_id}, "
            f"Open Queues={self.current_state_incoming_queue_ids})"
        )


class GlobalIntersection:
    def __init__(self, intersections, policies, all_queues_map, graph_G):
        self.all_intersections = []
        self.all_queues_map = all_queues_map
        if not isinstance(policies, abc.Mapping):
            policies = {id : policies for id in intersections}
        for intersection_id in intersections:
            intersection_instance = Intersection(intersection_id, policies[intersection_id], all_queues_map, graph_G)
            self.all_intersections.append(intersection_instance)

    def run_all_intersections_step(self, completed_cars_log):
        """Calls the step method on all instantiated Intersection objects."""
        for intersection in self.all_intersections:
            intersection.step(completed_cars_log)

    def __len__(self):
        total_cars_in_queues = 0
        for queue in self.all_queues_map.values():
            total_cars_in_queues += queue.count_cars()
        return total_cars_in_queues

    def __repr__(self):
        return "Instantiated Intersection objects:\n" + "\n".join(str(i) for i in self.all_intersections)

# ======================================================================================================================
# method to run the road network on one set of policies for so many time steps and return stats
# ======================================================================================================================

def run_simulation(G, num_timesteps, policies, generation_rates, wait_time_func=lambda S : 2 + 2*np.log(S), verbose=0):
  '''policies is a single generation rate or a dict mapping generator indices to generation rates'''
  all_queues_map = {}
  for u, v in G.edges():
      all_queues_map[(u, v)] = TrafficQueue((u, v), wait_time_func=wait_time_func)

  incoming_queues_adj_list = {}
  for u, v in G.edges():
      if v not in incoming_queues_adj_list:
          incoming_queues_adj_list[v] = {}
      incoming_queues_adj_list[v][u] = all_queues_map[(u, v)]

  intersections = []
  for node in G.nodes():
      if G.nodes[node]['type'] == 'n intersection':
          intersections.append(node)

  global_generator = GlobalGenerator(incoming_queues_adj_list, generation_rates=generation_rates)
  global_intersection = GlobalIntersection(intersections, policies, all_queues_map, G)
  completed_cars_log = []
  queue_fullness_history = [] # Re-initialize to ensure it's empty before simulation
  intersection_costs_history = []

  if verbose > 0:
      print(f"Starting simulation for {num_timesteps} timesteps...")

  for t in range(num_timesteps):
      if verbose > 0:
          print(f"\n--- Time Step {t+1} ---")

      # 1. Generators generate cars
      global_generator.run_all_generators()

      # 2. Intersections move cars
      global_intersection.run_all_intersections_step(completed_cars_log)

      # 3. Increment wait times for all cars in all queues
      increment_all_car_wait_times(all_queues_map)

      # 4. Record current queue states for analysis
      current_queue_state = {}
      for qid, queue_obj in all_queues_map.items():
          current_queue_state[qid] = queue_obj.count_cars()
      queue_fullness_history.append(current_queue_state)

      intersection_costs = {}
      for intersection in global_intersection.all_intersections:
          intersection_costs[intersection.intersection_id] = intersection.get_cost()
      intersection_costs_history.append(intersection_costs)

      if verbose > 0:
          print("Current Queue Fullness:")
          for qid, count in current_queue_state.items():
              print(f"  Queue {qid}: {count} cars")

  percent_cars_finished =  (len(completed_cars_log) / (len(global_intersection) + len(completed_cars_log))) if global_intersection else 1 # case where no cars are on road

  if verbose > 0:
      print("\nSimulation finished.")
      print("\n--- Queue Fullness History ---")
      for i, state in enumerate(queue_fullness_history):
          print(f"Time Step {i+1}: {state}")

      print("\n--- Completed Cars Log ---")
      for _ in completed_cars_log:
        print(_)

  return completed_cars_log, queue_fullness_history, intersection_costs_history, percent_cars_finished

# ======================================================================================================================
# method to compare different policies on the same network
# ======================================================================================================================

def run_multiple_simulations(G, num_timesteps, all_policies_list, generation_rates, verbose=0):
    # generation rates must be conisitant across all simulations to compare policies
    # policies is a list of policy profiles as described in run_simulation()
    results_by_policy = {}
    int_costs_by_policy = {}
    percent_cars_finished_by_policy = {}

    if verbose > 0: print(f"Running simulations for {num_timesteps} timesteps with different policies...")

    for policy in all_policies_list:
        policy_name = policy.__name__
        if verbose > 0: print(f"\n--- Running simulation with policy: {policy_name} ---")
        completed_cars_log, _, intersection_costs_history, percent_cars_finished = run_simulation(G, num_timesteps=num_timesteps, policies=policy, generation_rates=generation_rates, verbose=0)
        results_by_policy[policy_name] = completed_cars_log
        int_costs_by_policy[policy_name] = intersection_costs_history
        percent_cars_finished_by_policy[policy_name] = percent_cars_finished
        if verbose > 0: print(f"Simulation with {policy_name} completed. {len(completed_cars_log)} cars reached the destination.")

    if verbose > 0: print("All simulations completed.")
    if verbose > 0: print(f"Results stored for {len(results_by_policy)} policies: {list(results_by_policy.keys())}")

    grouped_travel_times = {}
    if verbose > 0: print("Grouping travel times by policy and generator...")
    for policy_name, completed_cars_log in results_by_policy.items():
        current_policy_data = defaultdict(list)
        for car_data in completed_cars_log:
            origin = car_data['origin']
            total_travel_time = car_data['total_travel_time']
            current_policy_data[origin].append(total_travel_time)
        grouped_travel_times[policy_name] = dict(current_policy_data)

    exp_int_costs_by_policy = {}
    for policy_name, intersection_costs_history in int_costs_by_policy.items():
        exp_int_costs = {}
        for int_costs_dict in intersection_costs_history:
            for int_id, cost in int_costs_dict.items():
                if int_id not in exp_int_costs:
                    exp_int_costs[int_id] = []
                exp_int_costs[int_id].append(cost)
        exp_int_costs_by_policy[policy_name] = exp_int_costs

    if verbose > 0:
        print("Travel times successfully grouped.")
        print(f"Grouped data available for policies: {list(grouped_travel_times.keys())}")
        print("Example structure for one policy (first generator): ")
        if grouped_travel_times:
            first_policy_name = list(grouped_travel_times.keys())[0]
            if grouped_travel_times[first_policy_name]:
                first_generator = list(grouped_travel_times[first_policy_name].keys())[0]
                print(f"  grouped_travel_times['{first_policy_name}']['{first_generator}'] = {grouped_travel_times[first_policy_name][first_generator][:5]}...")

    return grouped_travel_times, exp_int_costs_by_policy, percent_cars_finished_by_policy

