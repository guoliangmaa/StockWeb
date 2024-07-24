import datetime
import time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

t = datetime.datetime.now()


def init_parameters():
    global t
    t = datetime.datetime.now()


def get_time():
    init_parameters()
    print(t)


if __name__ == '__main__':
    # 创建调度器
    scheduler = BackgroundScheduler()

    # 使用 CronTrigger 创建每天 0 点 5 分钟执行的任务
    scheduler.add_job(get_time, CronTrigger(second=5))

    # 启动调度器
    scheduler.start()

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        # 关闭调度器
        scheduler.shutdown()
