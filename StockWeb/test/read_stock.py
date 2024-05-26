import tushare as ts

ts.set_token("2b029b986295fc98ff15e5dabfaa64cb48c5718d2268ec4d50319f90")
pro = ts.pro_api()

df = pro.daily(trade_date='20240508')
# df = pro.daily(ts_code="000001")
print(df)