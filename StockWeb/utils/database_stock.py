import io
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from StockWeb.utils.factory import get_mysql_engine

engin = get_mysql_engine(database="predict_stock")


def recommend_stock() -> list:
    sql = text(
        f'select id, `date`, stock_code, stock_name, data from recommend where `date`={datetime.today().strftime("%Y%m%d")}')
    result = []
    with engin.connect() as connection:
        res = connection.execute(sql)
        connection.commit()
        for row in res:
            id, _date, code, name, df = row
            csv_data = io.StringIO(df)
            df = pd.read_csv(csv_data)
            # print(row)
            df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
            df.fillna(0, inplace=True)
            mp = {
                "id": id,
                "date": _date,
                "code": code,
                "name": name,
                "df": df
            }
            print(mp)
            result.append(mp)

    return result


def stock_meta(stock_code: str) -> dict:
    """根据股票代码返回股票的详细信息"""
    sql = text(f"select * from stock_basic where ts_code like '%{stock_code}%'")
    with engin.connect() as connection:
        res = connection.execute(sql)
        connection.commit()
        result = [dict(row) for row in res.mappings()]
    return result[0]


if __name__ == '__main__':
    print(stock_meta("000001.sz"))
