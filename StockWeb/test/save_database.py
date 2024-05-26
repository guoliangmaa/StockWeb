import tushare as ts
from sqlalchemy import create_engine
import pandas as pd

# 设置Tushare的token
ts.set_token('2b029b986295fc98ff15e5dabfaa64cb48c5718d2268ec4d50319f90')
pro = ts.pro_api()

# 获取股票数据，例如每日数据
df = pro.daily(ts_code='000001.SZ')

# 数据库连接信息
user = 'root'
password = ''
host = '127.0.0.1'
port = '3306'
database = 'stock'

# 创建数据库引擎
engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}')

# 将DataFrame存储到MySQL数据库中的stock_data表
# df.to_sql('stock_data', engine, index=False, if_exists='append')
df.to_sql('stock_data', engine, index=False, if_exists='replace')

print("数据已成功存储到数据库")

query = "select * from stock_data"
res = pd.read_sql(query, engine)
print(res)
