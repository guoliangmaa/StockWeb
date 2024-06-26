from datetime import datetime, timedelta
import random

import pandas as pd
from pandas import DataFrame
from sqlalchemy import text
from StockWeb.utils.factory import get_tushare, get_mysql_engine
from StockWeb.models.lstm.train_and_predict import lstm_train_using_high_and_low, lstm_predict
from StockWeb.utils.Config import config
from StockWeb.utils.next_day import next_workday_str, recent_workday

ts = get_tushare()
today = datetime.today().strftime("%Y%m%d")
begin = (datetime.today() - timedelta(days=240)).strftime("%Y%m%d")

# 首先需要获取所有的股票代码
data = ts.stock_basic()


# print(data)

def run():
    stock_code_list = data["ts_code"].values
    stock_name_list = data["name"].values
    length = len(stock_code_list)
    # print(stock_code_list)

    res_json = []
    stock_count = 0
    unique_list = set()

    # 将基础数据存入数据库中
    engin = get_mysql_engine(database="predict_stock")
    data.to_sql(name="stock_basic", con=engin, if_exists="replace")

    while stock_count < 10:
        idx = random.randint(1, length)
        if idx not in unique_list:
            code = stock_code_list[idx]
            name = stock_name_list[idx]
            df = predict_future(code)
            if is_recommended_easy(df):
                # print(df)
                stock_count += 1
                print(f"股票 {code} 值得购买")
                res_json.append((today, code, name, df))
    sql = text(
        "insert into recommend (date, stock_code, stock_name, data) values (:date, :stock_code, :stock_name, :data)")
    sql_parameter = []
    for item in res_json:
        date, stock_code, stock_name, df = item
        df_string = df.to_csv(index=False)
        dat = {
            "date": date,
            "stock_code": stock_code,
            "stock_name": stock_name,
            "data": df_string
        }
        sql_parameter.append(dat)

    with engin.connect() as connection:
        # print(connection)
        connection.execute(sql, sql_parameter)
        connection.commit()
        # res = connection.execute("select * from recommend")


def predict_future(stock_code: str, future: int = 3) -> DataFrame:
    df: DataFrame = ts.daily(ts_code=stock_code, start_date=begin, end_date=today)
    _config = config()
    _config.stock_code = stock_code
    _config.epochs = 400
    _config.timestep = 10
    df = df.iloc[::-1].tail(120).reset_index(drop=True)

    lstm_train_using_high_and_low(_config, df)

    last_date = df.iloc[-1]["trade_date"]
    df.loc[:, "predict_high"] = -1
    df.loc[:, "predict_low"] = -1

    for i in range(int(len(df) / 2), len(df)):
        test_data = df.iloc[:i]  # 左闭右开 所以是前 i-1 天
        high, low = lstm_predict(_config, test_data)
        last_date = df.iloc[i]["trade_date"]
        df.at[i, "predict_high"] = high
        df.at[i, "predict_low"] = low

    for _ in range(future):
        print(last_date)
        test_data = df
        high, low = lstm_predict(_config, test_data)
        nxt_workday = next_workday_str(last_date)
        new_row = pd.DataFrame({
            "ts_code": [_config.stock_code],
            "high": [high],
            "low": [low],
            "close": [(high + low) / 2],
            "open": [(high + low) / 2],
            "trade_date": [nxt_workday],
            "predict_high": [high],
            "predict_low": [low]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        last_date = nxt_workday
    return df


def is_recommended(df: DataFrame, future: int = 3) -> bool:
    """苛刻版 必须预测一直走增 才返回true"""
    sub = df.tail(future + 1).reset_index(drop=True)
    for i in range(future):
        tdy = sub.iloc[i]["high"]
        nxt = sub.iloc[i + 1]["high"]
        # print(f"{sub.iloc[i]['trade_date']} 天的价格为 {tdy}, {sub.iloc[i + 1]['trade_date']}天的价格为 {nxt}")
        if not tdy > nxt:
            return False
    return True


def is_recommended_easy(df: DataFrame, future: int = 3) -> bool:
    """简易版 预测的最后的一天比当前的价格高 返回true"""
    sub = df.tail(future + 1).reset_index(drop=True)
    head = sub.iloc[0]["high"]
    end = sub.iloc[future]["high"]

    # print(f"当前价格{head} 预测未来{future}天价格 {end}")
    return end > head


if __name__ == '__main__':
    run()
