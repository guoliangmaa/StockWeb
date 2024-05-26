import tushare as ts
from sqlalchemy import create_engine
from tushare.pro.client import DataApi


def get_tushare() -> DataApi:
    # 设置Tushare的token
    ts.set_token('2b029b986295fc98ff15e5dabfaa64cb48c5718d2268ec4d50319f90')
    pro = ts.pro_api()
    return pro


def get_mysql_engine(
        user='root',
        password='',
        host='127.0.0.1',
        port='3306',
        database='stock',
):
    # 创建数据库引擎
    return create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}')


local_engine_instance = get_mysql_engine()
remote_engine_instance = get_mysql_engine(
    user='root',
    password='',
    host='127.0.0.1',
    port='3306',
    database='stock',
)
