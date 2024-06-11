from datetime import datetime, timedelta
from StockWeb.utils.factory import get_tushare

pro = get_tushare()


# 获取股票信息 并按照日期升序排序
def read_stock(stock_code, length=365 * 2) -> tuple:
    global df, new_df
    arr = (".SH", ".SZ", ".BJ")
    end_time = datetime.today()
    begin_time = end_time - timedelta(days=length)
    if not stock_code.endswith((".SH", ".SZ", ".BJ")):
        for item in arr:
            code = stock_code
            code = code + item
            # print(code)
            df = pro.daily(ts_code=code, start_date=begin_time.strftime("%Y%m%d"), end_date=end_time.strftime("%Y%m%d"))
            if not df.empty:
                # print(df)
                early_day = df["trade_date"][0]
                today = datetime.today().strftime("%Y%m%d")
                next_day = (datetime.today() + timedelta(days=1)).strftime("%Y%m%d")
                if early_day < today:
                    next_day = today
                print(next_day)
                df = df.head(length).iloc[::-1].reset_index(drop=True)  # 获取的数据是按照时间降序的 需要重排
                # df.to_csv(f"csv/{code}.csv", index=False)
                csv_df = df[["trade_date", "open", "close", "high", "low", "vol", "pct_chg"]]
                # csv_df.to_csv(f"csv/{code}_new.csv", index=False)
                predict_df = df[["open", "close", "high", "low", "vol", "pct_chg"]]

                return predict_df, df, code, next_day
