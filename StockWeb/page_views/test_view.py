from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView
import pandas as pd
from ..utils import read_stock
from ..tcn.predict_pth import predict
from ..tcn.Config import config
from ..tcn.test_stock_TCN_run import train_model as tcn_train


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
        timestep = request.GET.get("timestep", 50)  # 看多少天以前的数据

        self._config.epochs = self.epoch
        self._config.timestep = timestep
        self._config.stock_code = stock_code

        if model_name == "tcn":
            return self.model_tcn(self._config)

    def post(self, request):
        print("这是一个post请求")
        print(request)

        return Response(data=self.msg)

    def model_tcn(self, _config: config) -> Response:
        # 获得dataframe
        df, code = read_stock(_config.stock_code)
        # _config.data_path = f"csv/{code}_new.csv"

        if self.retrain:
            tcn_train(_config)

        res = predict(df)
        self.msg["data"] = df
        self.msg["predict"] = res
        return Response(data=self.msg)

    def model_cnn(self):
        pass

    def model_lstm(self) -> Response:
        pass
