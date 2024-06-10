import tushare as ts
from sqlalchemy import create_engine
from tushare.pro.client import DataApi


def get_tushare() -> DataApi:
    # 设置Tushare的token
    ts.set_token('a20f27bc10acf078a49505b86f815ab3563f10c3613b085b4063e00a')
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
