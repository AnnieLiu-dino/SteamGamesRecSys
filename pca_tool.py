from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import PCA
import matplotlib.pyplot as plt


# Apply PCA to reduce the vectors to 2 dimensions
def apply_pca(df_vect, k=2):
    pca = PCA(k=k, inputCol="features", outputCol="pcaFeatures")
    model_pca = pca.fit(df_vect)
    return model_pca.transform(df_vect).select("pcaFeatures")


# Get the index of each game ID
def get_game_indices(word_vectors):
    return {row["word"]: i for i, row in enumerate(word_vectors.collect())}


# Get the PCA result coordinates for specific games
def get_favorite_game_coords(favorite_games, game_indices, x, y):
    fav_x = []
    fav_y = []
    for game_id in favorite_games:
        if str(game_id) in game_indices:
            idx = game_indices[str(game_id)]
            fav_x.append(x[idx])
            fav_y.append(y[idx])
    return fav_x, fav_y


def plot_pca_by_user(
    user1_favorite_games, user2_favorite_games, word_vectors, spark, txt
):
    # word_vectors is a DataFrame with 'word':'vector'
    # Convert word_vectors to a format suitable for PCA
    data = [
        (Vectors.dense(x),)
        for x in word_vectors.select("vector").rdd.flatMap(lambda x: x).collect()
    ]
    df_vect = spark.createDataFrame(data, ["features"])
    print(f"vector_length: { len(df_vect.first()['features'])}")

    result_pca = apply_pca(df_vect)

    # Convert data for plotting
    x = []
    y = []
    for row in result_pca.collect():
        x.append(row["pcaFeatures"][0])
        y.append(row["pcaFeatures"][1])

    # Get the index of game IDs: {"game_id": index}
    game_indices = get_game_indices(word_vectors)

    # Get PCA results for the user's favorite games
    user1_x, user1_y = get_favorite_game_coords(
        user1_favorite_games, game_indices, x, y
    )
    user2_x, user2_y = get_favorite_game_coords(
        user2_favorite_games, game_indices, x, y
    )

    # Plot the PCA results for all games
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, color="lightblue", s=10, label="All Games")

    # Highlight specific games
    plt.scatter(user1_x, user1_y, color="red", s=10, label="User 1 Favorite Games")
    plt.scatter(user2_x, user2_y, color="blue", s=10, label="User 2 Favorite Games")

    # Add title and axis labels
    plt.title(f"PCA Projection of Game Vectors - {txt}")

    # First principal component
    plt.xlabel("Principal Component 1")
    # Second principal component
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()
