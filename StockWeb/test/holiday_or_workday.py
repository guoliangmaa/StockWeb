import datetime

# 判断 2018年4月30号 是不是节假日
from chinese_calendar import is_holiday, is_workday
april_last = datetime.date(2024, 4, 28)
print(is_holiday(april_last))
print(is_workday(april_last))

# 或者在判断的同时，获取节日名
import chinese_calendar as calendar  # 也可以这样 import
on_holiday, holiday_name = calendar.get_holiday_detail(april_last)
print(on_holiday)
print(holiday_name)

# 还能判断法定节假日是不是调休
import chinese_calendar
assert chinese_calendar.is_in_lieu(datetime.date(2006, 2, 1)) is False
assert chinese_calendar.is_in_lieu(datetime.date(2006, 2, 2)) is True

print("==========测试1=============")
from StockWeb.utils.next_day import next_workday
print(next_workday(datetime.datetime(year=2024, month=6, day=7)))

