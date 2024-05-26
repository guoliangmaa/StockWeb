from pandas import DataFrame
from .factory import get_mysql_engine
import pandas as pd

engine = get_mysql_engine()


def insert(table_name: str, df: DataFrame) -> None:
    df.to_sql(table_name, engine, index=False, if_exists="append")


def select(sql: str) -> DataFrame:
    return pd.read_sql(sql, engine)
