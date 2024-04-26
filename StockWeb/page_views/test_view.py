from pandas import DataFrame
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView
from ..utils import read_stock
from ..tcn.predict_pth import predict
from StockWeb.Config import config
from ..tcn.test_stock_TCN_run import train_model as tcn_train
from ..cnn.train import cnn_train
from ..cnn.predict import cnn_predict


class TestView(APIView):
    renderer_classes = [JSONRenderer]
    msg = {"msg": "success",
           "d": [],
           "predict": 0.0}
    retrain = True
    epoch = 20
    _config = config()

    def get(self, request):
        print(request)
        stock_code = request.GET.get("stock_code", "000001")  # 股票代码
        model_name = request.GET.get("model", "tcn")  # 模型选择
        length = request.GET.get("length", 365)  # 看多少天以前的数据

        self._config.epochs = self.epoch
        self._config.length = length
        self._config.stock_code = stock_code

        # 获得DataFrame
        df, origin_df, code = read_stock(self._config.stock_code, self._config.length)
        self._config.data_path = f"csv/{code}_new.csv"
        self._config.save_path = f"StockWeb/{model_name}/model.pth"
        if model_name == "tcn":
            return self.model_tcn(self._config, df, origin_df)
        elif model_name == "cnn":
            self._config.timestep = 10  # 原本代码设置为10
            return self.model_cnn(self._config, origin_df)
        elif model_name == "lstm":
            return self.model_lstm(self._config, origin_df)

    def post(self, request):
        print("这是一个post请求")
        print(request)

        return Response(data=self.msg)

    def model_tcn(self, _config: config, df: DataFrame, origin_df: DataFrame) -> Response:

        if self.retrain:
            tcn_train(_config)

        res = predict(df)
        self.msg["data"] = origin_df
        self.msg["predict"] = res
        return Response(data=self.msg)

    def model_cnn(self, _config: config, origin_df: DataFrame) -> Response:
        if self.retrain:
            cnn_train(_config, origin_df)

        res = cnn_predict(_config, origin_df)
        self.msg["predict"] = res
        self.msg["data"] = origin_df

        return Response(data=self.msg)

    def model_lstm(self, _config: config, origin_df: DataFrame) -> Response:

        return Response(data=self.msg)
