import tushare as ts
from datetime import datetime, timedelta

from pandas import DataFrame


def read_stock(stock_code, length=365 * 2) -> tuple:
    global df, new_df
    ts.set_token("a20f27bc10acf078a49505b86f815ab3563f10c3613b085b4063e00a")
    pro = ts.pro_api()
    arr = (".SH", ".SZ")
    end_time = datetime.today()
    begin_time = end_time - timedelta(days=length)
    if not stock_code.endswith((".SH", ".SZ")):
        for item in arr:
            code = stock_code
            code = code + item
            print(code)
            df = pro.daily(ts_code=code, start_date=begin_time.strftime("%Y%m%d"), end_date=end_time.strftime("%Y%m%d"))
            if not df.empty:
                print(df)
                df = df.head(length)[::-1]
                df.to_csv(f"csv/{code}.csv", index=False)
                csv_df = df[["trade_date", "open", "close", "high", "low", "vol", "pct_chg"]]
                csv_df.to_csv(f"csv/{code}_new.csv", index=False)
                predict_df = df[["open", "close", "high", "low", "vol", "pct_chg"]]

                return predict_df, df, code
