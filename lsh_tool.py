import faiss
import builtins
import numpy as np
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.sql.functions import col


# Convert map to two arrays: one for app IDs and one for their corresponding embeddings
def process_embeddings_map(embedding_map):
    embedding_dict_items = embedding_map.items()
    embeddings = [item for item in embedding_dict_items if len(item[1]) > 0]
    ids = [item[0] for item in embeddings]
    embeddings = [item[1] for item in embeddings]
    return ids, embeddings


class LSH:
    def __init__(self, game_embedding_map=None, nbits=256, bucketSize=50):
        if game_embedding_map is not None:
            self.initialize(game_embedding_map, nbits, bucketSize)

    def initialize(self, game_embedding_map, nbits, bucketSize):
        game_app_ids, game_embeddings = process_embeddings_map(game_embedding_map)
        self.app_ids = game_app_ids

        if game_embeddings:  # Ensure there are embeddings to process
            embedding_len = len(game_embeddings[0])
            # Initialize LSH with nbits and bucketSize
            self.lsh_instance = faiss.IndexLSH(embedding_len, nbits)
            self.lsh_instance.bucketSize = bucketSize
            self.lsh_instance.add(np.asarray(game_embeddings, dtype=np.float32))
            print(
                f"LSH index added {self.lsh_instance.ntotal} vectors with nbits={nbits} and bucketSize={bucketSize}"
            )

    def search(self, embedding, search_num=40):
        D, I = self.lsh_instance.search(
            np.asarray([embedding], dtype=np.float32), search_num
        )
        search_res = [self.app_ids[index] for index in I[0]]
        return search_res


def get_lsh_instance(game_embedding_map=None, nbits=256, bucketSize=50):
    return LSH(game_embedding_map, nbits, bucketSize)
