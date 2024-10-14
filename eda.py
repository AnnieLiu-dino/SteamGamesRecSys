import matplotlib.pyplot as plt

RATING_MAP = {
    "Overwhelmingly Positive": 9,
    "Very Positive": 8,
    "Positive": 7,
    "Mostly Positive": 6,
    "Mixed": 5,
    "Mostly Negative": 4,
    "Negative": 3,
    "Very Negative": 2,
    "Overwhelmingly Negative": 1,
}


def numeric_rating(game_df):
    game_df["numeric_rating"] = game_df["rating"].map(RATING_MAP)
    return game_df


def get_tags_map(game_info_df):
    tags_map = {}

    for index, row in game_info_df.iterrows():
        try:
            game_tags = [tag.strip() for tag in row["tags"]]
            for tag in game_tags:
                if tag not in tags_map:
                    tags_map[tag] = 0
                tags_map[tag] += 1
        except Exception as e:
            print(f"Error processing row {index}: {e}")

    sorted_tags_map = dict(
        sorted(tags_map.items(), key=lambda item: item[1], reverse=True)
    )
    return sorted_tags_map


def show_tag_dist(sorted_tags_map, top_n=50):
    labels = list(sorted_tags_map.keys())
    print("total tag number: ", len(labels))

    values = list(sorted_tags_map.values())

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(labels[:top_n], values[:top_n])
    plt.xticks(rotation=70, fontsize=9)
    plt.title("Top {} Tag Counts".format(top_n))
    plt.xlabel("Tags")
    plt.ylabel("Counts")
    plt.show()


def show_rating_dist(game_df):
    num_rating_type = game_df["rating"].nunique()

    plt.figure(figsize=(10, 6))
    game_df["numeric_rating"].plot.hist(
        bins=2 * num_rating_type, edgecolor="black", alpha=0.7
    )

    plt.title("Distribution of Numeric Ratings")
    plt.xlabel("Numeric Rating")
    plt.ylabel("Frequency")

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    print(game_df["numeric_rating"].describe())


def show_review_num_dist(game_df):
    rating_groupby = game_df.groupby("user_reviews").size()

    plt.figure(figsize=(12, 8))
    rating_groupby.plot.hist(bins=100, edgecolor="black", alpha=0.5)

    plt.title("Distribution of User Reviews")
    plt.xlabel("Number of User Reviews")
    plt.ylabel("Frequency")

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    print("Rating Group By:")
    print(rating_groupby.describe().round(2))


def show_user_rating_dist(review_df):
    user_rating_count = review_df.groupby("user_id").size()

    plt.figure(figsize=(12, 8))
    user_rating_count.plot.hist(bins=100, edgecolor="black", alpha=0.5)

    plt.title("Distribution of User Ratings")
    plt.xlabel("Number of Ratings per User")
    plt.ylabel("Frequency")

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    print("User Rating Count Description:")
    print(user_rating_count.describe().round(2))
