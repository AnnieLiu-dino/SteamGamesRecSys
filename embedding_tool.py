from pyspark.sql import functions as F
from pyspark.ml.feature import Word2Vec, Word2VecModel
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pool.game_matrix import sequence2matrix, combine_matrix
import dill
import numpy as np
import os
import tqdm
from bisect import bisect


def parallel_game_matrix_processing(game_sequences):
    """Process game sequences in parallel to generate the game matrix."""
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results = list(executor.map(sequence2matrix, game_sequences))
        game_matrix = combine_matrix(results)
    return game_matrix


def compute_trans_probs(games_chain):
    """Compute transition probabilities."""
    apps_id = list(games_chain.keys())
    total_weight = sum(games_chain.values())
    probs = [games_chain[id] / total_weight for id in apps_id]
    return {"apps_chain": apps_id, "probs": probs}


def build_trans_prob_matrix(game_matrix):
    """Build transition probability matrix."""
    trans_prob_matrix = {}
    for app_id, games_chain in game_matrix.items():
        result = compute_trans_probs(games_chain)
        trans_prob_matrix[app_id] = result
    return trans_prob_matrix


def build_entry_games_probs(trans_prob_matrix, game_matrix):
    """Compute entry probabilities."""
    all_app_ids = list(trans_prob_matrix.keys())
    games_weight_map = {
        app_id: sum(game_matrix[app_id].values()) for app_id in all_app_ids
    }
    all_games_weight = sum(games_weight_map.values())
    entry_game_probs = [
        games_weight_map[app_id] / all_games_weight for app_id in all_app_ids
    ]
    return all_app_ids, entry_game_probs


def precompute_cumulative_probs(trans_prob_matrix):
    cumulative_prob_matrix = {}
    for node, data in trans_prob_matrix.items():
        cumulative_probs = np.cumsum(data["probs"])
        cumulative_prob_matrix[node] = (data["apps_chain"], cumulative_probs)
    return cumulative_prob_matrix


def fast_choice(app_ids, cumulative_probs, rng):
    idx = bisect(cumulative_probs, rng.random())
    return app_ids[max(0, idx - 1)]  # max to handle edge case where idx is 0


# Ensure all functions are in the global scope
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
            idx = np.searchsorted(cumulative_probs, rng.random())
            cur_app_id = app_ids[max(0, idx - 1)]
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


def deep_walks(
    sampling_times, walk_depth, all_app_ids, all_app_probs, trans_prob_matrix
):
    cumulative_prob_matrix = precompute_cumulative_probs(trans_prob_matrix)
    num_cores = mp.cpu_count()
    seeds = np.random.randint(0, 2**32 - 1, size=sampling_times)

    # Reduce the number of tasks and increase the number of walks per task
    tasks = [
        (100, walk_depth, all_app_ids, all_app_probs, cumulative_prob_matrix, seed)
        for seed in seeds[: sampling_times // 100]
    ]

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm.tqdm(executor.map(random_walks, tasks), total=len(tasks)))

    flat_results = [item for sublist in results for item in sublist]
    return flat_results


def build_game_sequences(user_play_sequence_df):
    """Build game sequences from user behavior DataFrame."""
    game_sequences = [row["app_ids"] for row in user_play_sequence_df.collect()]
    return game_sequences
