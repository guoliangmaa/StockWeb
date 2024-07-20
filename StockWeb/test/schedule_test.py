from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import time

def my_task():
    print(f"任务执行中... 当前时间: {datetime.now()}")

# 创建调度器
scheduler = BackgroundScheduler()

# 使用 CronTrigger 创建每天 0 点 5 分钟执行的任务
scheduler.add_job(my_task, CronTrigger(hour=19, minute=13))

# 启动调度器
scheduler.start()

try:
    while True:
        time.sleep(60)
except (KeyboardInterrupt, SystemExit):
    # 关闭调度器
    scheduler.shutdown()
