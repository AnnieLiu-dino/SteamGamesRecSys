import numpy as np
import concurrent.futures
import multiprocessing as mp


# Weighted random walk, selecting based on probability weights
# TODO: Consider: Nodes with very small weights have a very low probability of being sampled; Restarting random walks: Introduce the concept of restart in random walks, where there is a certain probability at each step to jump to any node, increasing the chances of all nodes being visited in the network.
def single_random_walk(
    walk_depth,
    all_app_ids,
    all_app_probs,
    cumulative_prob_matrix,
    rng,
    restart_prob=0.15,
):
    start_app_id = rng.choice(all_app_ids, p=all_app_probs)
    walk_path = [start_app_id]
    cur_app_id = start_app_id

    for _ in range(walk_depth):
        if rng.random() < restart_prob:
            cur_app_id = start_app_id  # Restart
        else:
            app_ids, cumulative_probs = cumulative_prob_matrix[cur_app_id]
            cur_app_id = fast_choice(app_ids, cumulative_probs, rng)
        walk_path.append(cur_app_id)

    return walk_path


def random_walks(params):
    num_walks, walk_depth, all_app_ids, all_app_probs, cumulative_prob_matrix, seed = (
        params
    )
    rng = np.random.default_rng(seed)
    return [
        single_random_walk(
            walk_depth, all_app_ids, all_app_probs, cumulative_prob_matrix, rng
        )
        for _ in range(num_walks)
    ]
