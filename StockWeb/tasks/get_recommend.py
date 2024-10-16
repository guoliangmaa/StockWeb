import platform

if platform.system() == "Linux":
    import sys

    sys.path.append('/home/mgl/StockWeb')
from datetime import datetime, timedelta
import random
import pandas as pd
from pandas import DataFrame
from sqlalchemy import text
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import time
from StockWeb.utils.factory import get_tushare, get_mysql_engine
from StockWeb.models.lstm.train_and_predict import lstm_train_using_high_and_low, lstm_predict
from StockWeb.utils.Config import config
from StockWeb.utils.next_day import next_workday_str, recent_workday
from StockWeb.utils.mail_util import send_email

ts = get_tushare()
# 将基础数据存入数据库中
engin = get_mysql_engine(database="predict_stock")

today = datetime.today().strftime("%Y%m%d")
begin = (datetime.today() - timedelta(days=240)).strftime("%Y%m%d")

# 首先需要获取所有的股票代码
data = ts.stock_basic()


def init_parameters():
    global today, begin, data
    today = datetime.today().strftime("%Y%m%d")
    begin = (datetime.today() - timedelta(days=240)).strftime("%Y%m%d")

    # 首先需要获取所有的股票代码
    data = ts.stock_basic()


def recommend_stock():
    stock_code_list = data["ts_code"].values
    stock_name_list = data["name"].values
    length = len(stock_code_list)
    # print(stock_code_list)

    res_json = []
    stock_count = 0
    unique_list = set()

    with engin.connect() as connection:
        trans = connection.begin()
        try:
            data.to_sql(name="stock_basic", con=connection, if_exists="replace")
            trans.commit()
        except Exception as e:
            trans.rollback()
            print(f"Error in saving stock basic to database: {e}")

    while stock_count < 10:
        idx = random.randint(1, length)
        if idx not in unique_list:
            code = stock_code_list[idx]
            name = stock_name_list[idx]
            try:
                df = predict_future(code)
            except Exception:
                continue
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
        trans = connection.begin()
        try:
            connection.execute(sql, sql_parameter)
            trans.commit()
        except Exception as e:
            trans.rollback()
            print(f"Error in saving recommend stock to database: {e}")
        finally:
            connection.close()


def warning_stock():
    stock_list = ["002142", "600036", "600000", "601020", "003035", "600008", "600016", "600225", "600598", "600751",
                  "000547", "000593", "000848", "000063", "000066", "000895", "000001", "000008", "000019", "000062"]
    # 首先我们得先从 dataframe 中找到股票的数据信息
    stock_list_meta = []
    for stock in stock_list:
        res = data[data['symbol'] == stock]
        stock_list_meta.append({
            "code": res['ts_code'].values[0],
            "name": res['name'].values[0]
        })

    res_json = []
    for item in stock_list_meta:
        df = predict_future(item["code"])
        res_json.append((today, item["code"], item["name"], df, 1 if is_down_easy(df) else 0))

    sql = text(
        "insert into warning (date, stock_code, stock_name, data, warning) values (:date, :stock_code, :stock_name, :data, :warning)")
    sql_parameter = []
    for item in res_json:
        date, stock_code, stock_name, df, warning = item
        df_string = df.to_csv(index=False)
        dat = {
            "date": date,
            "stock_code": stock_code,
            "stock_name": stock_name,
            "data": df_string,
            "warning": warning
        }
        sql_parameter.append(dat)

    with engin.connect() as connection:
        trans = connection.begin()
        try:
            connection.execute(sql, sql_parameter)
            trans.commit()
        except Exception as e:
            trans.rollback()
            print(f"Error in saving warning stock to database: {e}")
        finally:
            connection.close()
    print(stock_list_meta)


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


def is_down_easy(df: DataFrame, future: int = 3) -> bool:
    """简易版 预测的最后的一天比当前的价格降低 返回true"""
    sub = df.tail(future + 1).reset_index(drop=True)
    head = sub.iloc[0]["high"]
    end = sub.iloc[future]["high"]

    # print(f"当前价格{head} 预测未来{future}天价格 {end}")
    return end < head


def task():
    print(f"{datetime.now()} 任务开始")
    init_parameters()
    recommend_stock()
    warning_stock()
    send_email(f"{today}股票信息")
    print(f"{datetime.now()} 任务结束")


if __name__ == '__main__':
    # 创建调度器
    scheduler = BackgroundScheduler()
    # 使用 CronTrigger 创建每天 0 点 30 分钟执行的任务
    scheduler.add_job(task, CronTrigger(hour=0, minute=30))
    # 启动调度器
    scheduler.start()

    try:
        while True:
            time.sleep(10)
    except (KeyboardInterrupt, SystemExit):
        # 关闭调度器
        scheduler.shutdown()