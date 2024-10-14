import pandas as pd
import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import os
import sys
import dill

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def init_spark_session(app_name = 'game_rs'):
    return SparkSession.builder.appName(app_name)\
        .config("spark.executor.memory", "32g") \
        .config("spark.driver.memory", "32g") \
        .config("spark.executor.cores", 12) \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.driver.maxResultSize", "12g") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()


def load_all_df(folder_path = './'):
    game_df = pd.read_csv(f'{folder_path}games.csv')
    review_df = pd.read_csv(f'{folder_path}recommendations.csv')
    user_df = pd.read_csv(f'{folder_path}users.csv', index_col='user_id')
    game_info_df = pd.read_json(f'{folder_path}/games_metadata.json', lines=True)
    return game_df, review_df, user_df, game_info_df


def load_spark_df(spark, folder_path, file_name):
    spark_df = spark.read.parquet(f"{folder_path}{file_name}")
    return spark_df

    
def search_game_info(df, app_id):
    matching_rows = df.loc[df['app_id'] == app_id]
    return matching_rows
    
def load_dill_file(filepath):
    """从dill文件中加载数据."""
    with open(filepath, 'rb') as f:
        data = dill.load(f)
    return data

def save_to_dill(data, file_name, folder_path='./saved_dill'):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = os.path.join(folder_path, file_name)
    try:
        with open(folder_path, 'wb') as f:
            dill.dump(data, f)
        print(f"Data successfully saved to {folder_path}")
    except Exception as e:
        print(f"Failed to save data: {e}")  

def load_all_dill_as_dict(folder_path):
    vars_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.dill'):
            var_name = os.path.splitext(filename)[0]
            with open(os.path.join(folder_path, filename), 'rb') as file:
                vars_dict[var_name] = dill.load(file)
    return vars_dict

