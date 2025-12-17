def all_open_policy(intersection):
    """Keeps all incoming queues open for the intersection."""
    intersection.current_state_incoming_queue_ids = list(intersection.incoming_queues_ids)

def cycle_policy(intersection):
    """Applies a simple cycling policy to determine which incoming queue is open."""
    if not intersection.incoming_queues_ids:
        intersection.current_state_incoming_queue_ids = []
        return

    # Simple cycling policy: open one incoming queue at a time
    queue_to_open_id = intersection.incoming_queues_ids[intersection.policy_step_counter % len(intersection.incoming_queues_ids)]
    intersection.current_state_incoming_queue_ids = [queue_to_open_id]

def greedy_longest_current_wait_policy(intersection):
    """Opens the incoming queue whose leading car has the longest current wait time."""
    if not intersection.incoming_queues_ids:
        intersection.current_state_incoming_queue_ids = []
        return

    longest_wait_time = -1
    queue_to_open_id = None

    for qid in intersection.incoming_queues_ids:
        traffic_queue = intersection.all_queues_map[qid]
        if not traffic_queue.is_empty():
            # We can't use peek_top_car_wait_time directly since it just gives wait_time,
            # but we need to compare across queues. Let's get the car object itself.
            top_car = traffic_queue.get_cars_in_queue()[0]
            if top_car.wait_time > longest_wait_time:
                longest_wait_time = top_car.wait_time
                queue_to_open_id = qid

    if queue_to_open_id is not None:
        intersection.current_state_incoming_queue_ids = [queue_to_open_id]
    else:
        intersection.current_state_incoming_queue_ids = [] # No cars in any incoming queue

def longest_current_wait_policy(intersection):
    """Opens the incoming queue containing the car with longest current wait time."""
    if not intersection.incoming_queues_ids:
        intersection.current_state_incoming_queue_ids = []
        return

    longest_wait_time = -1
    queue_to_open_id = None

    for qid in intersection.incoming_queues_ids:
        traffic_queue = intersection.all_queues_map[qid]
        if not traffic_queue.is_empty():
            # We can't use peek_top_car_wait_time directly since it just gives wait_time,
            # but we need to compare across queues. Let's get the car object itself.
            top_car_wait_time = max([car.wait_time for car in traffic_queue.get_cars_in_queue()])
            if top_car_wait_time > longest_wait_time:
                longest_wait_time = top_car_wait_time
                queue_to_open_id = qid

    if queue_to_open_id is not None:
        intersection.current_state_incoming_queue_ids = [queue_to_open_id]
    else:
        intersection.current_state_incoming_queue_ids = [] # No cars in any incoming queue

def longest_cumulative_wait_policy(intersection):
    """Opens the incoming queue whose leading car has the longest total travel time."""
    if not intersection.incoming_queues_ids:
        intersection.current_state_incoming_queue_ids = []
        return

    longest_total_travel_time = -1
    queue_to_open_id = None

    for qid in intersection.incoming_queues_ids:
        traffic_queue = intersection.all_queues_map[qid]
        if not traffic_queue.is_empty():
            top_car = traffic_queue.get_cars_in_queue()[0]
            if top_car.total_travel_time > longest_total_travel_time:
                longest_total_travel_time = top_car.total_travel_time
                queue_to_open_id = qid

    if queue_to_open_id is not None:
        intersection.current_state_incoming_queue_ids = [queue_to_open_id]
    else:
        intersection.current_state_incoming_queue_ids = [] # No cars in any incoming queue