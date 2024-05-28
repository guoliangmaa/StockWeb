import time
from datetime import datetime, timedelta

from pandas import DataFrame

from StockWeb.utils.factory import get_tushare as tushare
from StockWeb.utils.read_stock import read_stock
from StockWeb.utils.database import insert

ts = tushare()

# 首先获取所有的股票代码
df: DataFrame = ts.daily(trade_date=datetime.today().strftime("%Y%m%d"))
print(df)


def get_stock_specific_day(date: datetime) -> DataFrame:
    """
    获取特定天数所有的股票信息
    :param date 日期 datetime类型
    """
    str_date = date.strftime("%Y%m%d")
    return ts.daily(trade_date=str_date)


def get_stock_previous(previous: int) -> None:
    """
    获取历史股票数据(几天以前的) 并存到数据库中
    一般就调用一次 作为项目最开始的数据 以后只需要调用当天的数据即可
    :param previous 几天以前的
    """
    df = None
    for i in range(previous):
        df = get_stock_specific_day(datetime.today() - timedelta(days=i))
        if not df.empty:
            break

    for row in df.itertuples():
        _ = ts.daily(ts_code=row.ts_code,
                     start_date=(datetime.today() - timedelta(days=previous)).strftime("%Y%m%d"),
                     end_date=datetime.today().strftime("%Y%m%d"))
        insert(row.ts_code, _)
        time.sleep(120 / 1000)  # 一分钟最多500次 60000 / 500


if __name__ == '__main__':
    get_stock_previous(31 * 2)
