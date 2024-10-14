from collections import defaultdict, Counter

def sequence2matrix(sequence):
    local_matrix = defaultdict(Counter)
    user_play_game_num = len(sequence)
    for x in range(user_play_game_num):
        for y in range(x + 1, user_play_game_num):
            game_a_id = sequence[x]
            game_b_id = sequence[y]
            if game_a_id != game_b_id:
                local_matrix[game_a_id][game_b_id] += 1
                local_matrix[game_b_id][game_a_id] += 1
    return local_matrix

def combine_matrix(results):
    game_matrix = defaultdict(Counter)
    for result in results:
        for key, inner_map in result.items():
            for inner_key, count in inner_map.items():
                game_matrix[key][inner_key] += count
    return game_matrix