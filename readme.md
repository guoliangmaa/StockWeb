## 安装环境

> pip install Django==4.2.11  
> pytorch stable 2.3.0 (cuda11.8)  
> pip install pandas  
> pip install djangorestframework  
> pip install tushare (股票数据获取)  
> pip install scikit-learn  
> pip install matplotlib  
> pip install yfinance  
> pip install sqlalchemy pymysql  
> pip install chinesecalendar (应该每年更新该包 11月前后国务院发布通知)   
> pip install edge-tts   
> 



---
服务器部署命令
> nohup python manage.py runserver 0.0.0.0:8080 &> output.log &
> 
---
接口文档
- /api/test 单只股票预测 未来三天(最大值和最小值)
  - 参数1: stock_code 例如 000001(单纯数字组成的字符串 其他格式未做适配)
- /api/stock/recommend 推荐10支建议买入的股票 每日推荐
  - 无参数
- /api/stock/tts 语音转文字接口
  - 参数:text 文本
- /api/stock/warning 预警股票信息(未来三天会跌)
  - 无参数


> 注: 每天开盘前运行 StockWeb/tasks/get_recommend.py 以获得当日推荐