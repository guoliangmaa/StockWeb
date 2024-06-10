from pandas import DataFrame
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView
from ..utils.read_stock import read_stock
from StockWeb.models.tcn.predict_pth import predict
from StockWeb.utils.Config import config
from StockWeb.models.tcn.test_stock_TCN_run import train_model as tcn_train
from StockWeb.models.cnn.train import cnn_train
from StockWeb.models.cnn.predict import cnn_predict
# from StockWeb.models.lstm.train import lstm_train
# from StockWeb.models.lstm.predict import lstm_predict
from StockWeb.models.lstm.train_and_predict import lstm_train_using_high_and_low
from StockWeb.models.lstm.train_and_predict import lstm_predict
from StockWeb.utils.database import insert, select
from StockWeb.utils.next_day import next_workday, next_workday_str


class TestView(APIView):
    renderer_classes = [JSONRenderer]
    msg = {"msg": "success",
           "d": [],
           "predict": 0.0}
    retrain = True
    epoch = 200
    _config = config()
    history: DataFrame | None = None

    def get(self, request):
        print(request)
        stock_code = request.GET.get("stock_code", "000001")  # 股票代码
        # model_name = request.GET.get("model", "tcn").lower()  # 模型选择 默认都小写
        model_name = "lstm"  # 模型选择 默认为lstm
        # length = request.GET.get("length", 30 * 2)  # 看多少天以前的数据
        length = 60  # 看多少天以前的数据

        self._config.epochs = self.epoch
        self._config.length = int(length) * 4  # 为了方便预测历史的数据
        self._config.stock_code = stock_code

        # 获得DataFrame
        df, origin_df, code, next_day = read_stock(self._config.stock_code, self._config.length)
        self._config.data_path = f"csv/{code}_new.csv"
        self._config.save_path = f"StockWeb/models/{model_name}/model.pth"
        self._config.next_day = next_day
        try:
            self.history = select(f"select * from {code} order by trade_date desc")
        except Exception:
            self.history = None  # 为了防止多次请求导致数据混乱
            print("数据库中未存在历史数据")

        if model_name == "tcn":
            self._config.timestep = 50
            return self.model_tcn(self._config, df, origin_df)
        elif model_name == "cnn":
            self._config.timestep = 10  # 原本代码设置为10
            return self.model_cnn(self._config, origin_df)
        elif model_name == "lstm":
            self._config.timestep = 10  # 原本代码设置为10
            # return self.model_lstm(self._config, origin_df)
            return self.model_lstm_enhanced(self._config, origin_df)

    def post(self, request):
        print("这是一个post请求")
        print(request)

        return Response(data=self.msg)

    def model_tcn(self, _config: config, df: DataFrame, origin_df: DataFrame) -> Response:

        if self.retrain:
            tcn_train(_config)

        res = predict(df, _config)
        self.msg["data"] = origin_df
        self.msg["predict"] = res.item()
        self.msg["next_day"] = _config.next_day
        response = Response(data=self.msg)
        response['Access-Control-Allow-Origin'] = "*"
        return response

    def model_cnn(self, _config: config, origin_df: DataFrame) -> Response:
        if self.retrain:
            cnn_train(_config, origin_df)

        res = cnn_predict(_config, origin_df)
        self.msg["predict"] = res.item()
        self.msg["data"] = origin_df
        self.msg["next_day"] = _config.next_day
        response = Response(data=self.msg)
        response['Access-Control-Allow-Origin'] = "*"
        return response

    def model_lstm(self, _config: config, origin_df: DataFrame) -> Response:
        if self.retrain:
            # lstm_train(_config, origin_df)
            lstm_train_using_high_and_low(_config, origin_df)

        predict_high, predict_low = lstm_predict(_config, origin_df)

        self.msg["predict_high"] = predict_high.item()
        self.msg["predict_low"] = predict_low.item()
        self.msg["predict"] = (predict_high.item() + predict_low.item()) / 2
        self.msg["data"] = origin_df
        self.msg["next_day"] = _config.next_day
        response = Response(data=self.msg)
        response['Access-Control-Allow-Origin'] = "*"
        return response

    def model_lstm_enhanced(self, _config: config, origin_df: DataFrame) -> Response:
        append_date = []  # 需要更新到数据库中的数据 以时间为线索
        if self.history is None or self.history.empty:
            # 股票信息是按照时间升序排列的(从旧到新) 所以最新数据在尾部
            self.history = origin_df.tail(120)
            self.history.loc[:, "predict_high"] = -1
            self.history.loc[:, "predict_low"] = -1
        else:
            # 需要对比 数据库中的数据(时间降序) 和 处理过的tushare数据(时间升序) 的区别
            self.history = self.history[::-1]
            # 如果获取的最新数据 比 数据库中的新
            for row in origin_df.itertuples():
                if self.history["trade_date"][-1] < row.ts_code:
                    new_row = {}
                    new_row["ts_code"] = row.ts_code
                    new_row["trade_date"] = row.trade_date
                    new_row["open"] = row.open
                    new_row["close"] = row.close
                    new_row["high"] = row.high
                    new_row["low"] = row.low
                    new_row["pre_close"] = row.pre_close
                    new_row["change"] = row.change
                    new_row["pct_chg"] = row.pct_chg
                    new_row["vol"] = row.vol
                    new_row["amount"] = row.amount

                    new_row["predict_high"] = -1
                    new_row["predict_low"] = -1
                    self.history.append(new_row)

        if self.retrain:
            lstm_train_using_high_and_low(_config, self.history[["high", "low"]])

        # 这个循环可以预测 历史 和 未来一天 的股票
        arr = []
        for i in range(61, 121):
            test_data = self.history.head(i)
            last_date = test_data.tail(1)["trade_date"].values[0]
            high, low = lstm_predict(_config, test_data)
            # 格式为 当天日期 下一个非工作日的最高价和最低价
            arr.append([last_date, high, low])
            print(f"当前日期 {last_date}, 预测下一天最高价 {high}, 最低价 {low}")
            nxt_workday = next_workday_str(last_date)
            print(f"下一个工作日为 {nxt_workday}")
            row_index = self.history[self.history["trade_date"] == nxt_workday].index[0]
            print(f"查找到的行索引为 {row_index}")
            self.history.loc[row_index, "predict_high"] = high
            self.history.loc[row_index, "predict_low"] = low
        print(self.history)
        print(arr)
        response = Response(data=self.msg)
        response['Access-Control-Allow-Origin'] = "*"
        return response
