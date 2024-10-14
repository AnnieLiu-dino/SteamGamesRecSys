from pyspark.sql.functions import when, col, udf
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
import builtins
from pyspark.sql.functions import udf, col, explode, trim, lit
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list, max

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


# Process boolean columns by converting them to integers (0 or 1)
def process_boolean_col(df, col_name):
    return df.withColumn(col_name, when(col(col_name) == True, 1).otherwise(0))


def process_game_df(spark_game_df, spark_game_info_df):
    spark_completed_game_df = (
        spark_game_df.join(spark_game_info_df, on="app_id", how="inner")
        .withColumnRenamed("date_release", "app_release_date")
        .withColumn(
            "app_release_date", F.to_date(col("app_release_date"), "yyyy-MM-dd")
        )
        .withColumn(
            "app_release_ts", F.unix_timestamp(F.col("app_release_date"), "yyyy-MM-dd")
        )
        .withColumn("price_diff", col("price_original") - col("price_final"))
    )

    # Convert 'linux', 'mac', 'win', 'steam_deck' boolean columns to numeric
    boolean_cols = ["linux", "mac", "win", "steam_deck"]
    for col_name in boolean_cols:
        spark_completed_game_df = process_boolean_col(spark_completed_game_df, col_name)

    # Map ratings to numeric values
    @udf(IntegerType())
    def map_rating(rating):
        return RATING_MAP.get(rating, None)

    spark_completed_game_df = spark_completed_game_df.withColumn(
        "numeric_rating", map_rating(col("rating"))
    )
    return spark_completed_game_df


def process_review_df(spark_review_df):

    spark_review_df = (
        spark_review_df.withColumnRenamed("date", "review_date")
        .withColumn("review_date", F.to_date(col("review_date"), "yyyy-MM-dd"))
        .withColumn("review_ts", F.unix_timestamp(F.col("review_date"), "yyyy-MM-dd"))
        .withColumn("label", when(col("is_recommended") == True, 1).otherwise(0))
    )
    return spark_review_df


def generate_interactions_df(spark_review_df, spark_completed_game_df):
    selected_game_features_df = spark_completed_game_df.select(
        "app_id",
        "app_release_ts",
        "win",
        "mac",
        "linux",
        "numeric_rating",
        "positive_ratio",
        "user_reviews",
        "price_diff",
        "discount",
        "steam_deck",
        "tags",
    )

    user_item_interactions_df = spark_review_df.join(
        selected_game_features_df, on=["app_id"], how="left"
    )
    user_item_interactions_df = user_item_interactions_df.drop("review_id")
    return user_item_interactions_df


@F.udf(FloatType())
def extract_float(l):
    v = builtins.round(l[0], 2)
    return float(v)


# Min-Max scaling for numeric features to a 0-1 range with 2 decimal places
class NumMinMaxScaler:
    def __init__(self, cols):
        self.cols = cols
        self.pipeline = self.build_pipeline(cols)

    def build_pipeline(self, cols):
        pipelines = [self.build_scaler_stage(col) for col in cols]
        return Pipeline(stages=pipelines)

    def build_scaler_stage(self, col):
        output_col = f"{col}_scaled"
        vec_assembler = VectorAssembler(
            inputCols=[col], outputCol=f"{col}_vec", handleInvalid="keep"
        )
        scaler_instance = MinMaxScaler(inputCol=f"{col}_vec", outputCol=output_col)
        pipeline = Pipeline(stages=[vec_assembler, scaler_instance])
        return pipeline

    def fit(self, df):
        self.model = self.pipeline.fit(df)

    def transform(self, df):
        result = self.model.transform(df)

        for col in self.cols:
            output_col = f"{col}_scaled"
            result = result.drop(f"{col}_vec").withColumn(
                output_col, extract_float(F.col(output_col))
            )

        return result


def encode_tags(mapping_dict):
    @udf(returnType="array<int>")
    def encode_tags_udf(tags, max_tag_index):
        tags = tags or []
        gen_vec = list(set(mapping_dict.value.get(gen) for gen in tags))
        fill = np.ones(len(gen_vec), dtype=np.int32)
        sorted_index = np.sort(gen_vec)
        multihot_vec = SparseVector(max_tag_index + 1, sorted_index, fill)
        return multihot_vec.toArray().astype(np.int32).tolist()

    return encode_tags_udf


class TagEncoder:
    def __init__(self, colname):
        self.colname = colname

    def fit(self, df):
        exploded_df = df.withColumn("tag_item", explode(col(self.colname)))
        tag_stringIndexer = StringIndexer(inputCol="tag_item", outputCol="tag_index")
        indexer_model = tag_stringIndexer.fit(exploded_df)

        tags_info = spark.createDataFrame(
            [{"tag_item": g} for g in indexer_model.labels]
        )
        mapping_df = indexer_model.transform(tags_info).collect()
        mapping_dict = {row.tag_item: int(row.tag_index) for row in mapping_df}
        self.max_tag_index = builtins.max(mapping_dict.values())
        self.broadcasted = spark.sparkContext.broadcast(mapping_dict)

    @udf(returnType="array<int>")
    def encode_tags_udf(self, tags, max_tag_index):
        tags = tags or []
        tag_vec = list(set(self.broadcasted.value.get(tag) for tag in tags))
        # [...all 0]
        game_tags_arr = np.ones(len(tag_vec), dtype=np.int32)
        sorted_tag_index_arr = np.sort(tag_vec)
        # length = max_tag_index + 1,
        multihot_vec = SparseVector(
            max_tag_index + 1, sorted_tag_index_arr, game_tags_arr
        )
        return multihot_vec.toArray().astype(np.int32).tolist()

    def transform(self, df):
        return df.withColumn(
            f"{self.colname}_multihot",
            self.tags_encode_fun(col(self.colname), lit(self.max_tag_index)),
        )
