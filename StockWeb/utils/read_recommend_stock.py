import io
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from StockWeb.utils.factory import get_mysql_engine


def recommend_stock() -> list:
    engin = get_mysql_engine(database="predict_stock")
    sql = text(f'select id, `date`, stock_code, data from recommend where `date`={datetime.today().strftime("%Y%m%d")}')
    result = []
    with engin.connect() as connection:
        res = connection.execute(sql)
        connection.commit()
        for row in res:
            id, _date, code, df = row
            csv_data = io.StringIO(df)
            df = pd.read_csv(csv_data)
            # print(row)
            df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
            df.fillna(0, inplace=True)
            mp = {
                "id": id,
                "date": _date,
                "code": code,
                "df": df
            }
            print(mp)
            result.append(mp)

    return result
