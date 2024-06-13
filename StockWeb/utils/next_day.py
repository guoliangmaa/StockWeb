from datetime import datetime, timedelta
from chinese_calendar import is_holiday, is_workday


def next_workday(time: datetime | str) -> datetime:
    if isinstance(time, str):
        time = datetime.strptime(time, "%Y%m%d")

    while True:
        time = time + timedelta(days=1)
        if is_workday(time) and not is_weekend(time):
            break
    return time


def next_workday_str(time: datetime | str, f="%Y%m%d") -> str:
    return next_workday(time).strftime(f)


def is_weekend(date: datetime):
    # 获取星期几，返回0（周一）到6（周日）
    day_of_week = date.weekday()
    # 判断是否是周六（5）或周日（6）
    return day_of_week == 5 or day_of_week == 6


def recent_workday(time: datetime | str, f="%Y%m%d") -> str:
    """返回最近一个工作日"""
    if isinstance(time, str):
        time = datetime.strptime(time, f)
    while is_holiday(time) or is_weekend(time):
        time = time - timedelta(days=1)

    return time.strftime(f)