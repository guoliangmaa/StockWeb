from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView
import pandas as pd
from ..utils import read_stock
from ..tcn.predict_pth import predict


class TestView(APIView):
    renderer_classes = [JSONRenderer]
    msg = {"msg": "success",
           "d": [],
           "predict": 0.0}
    retrain = False

    def get(self, request):
        print(request)
        stock_code = request.GET.get('stock_code', '000001')
        # stock_code = "000001"
        if self.retrain:
            pass
        else:
            pass
        # 获得dataframe
        df = read_stock(stock_code)
        res = predict(df)
        self.msg["data"] = df
        self.msg["predict"] = res
        return Response(data=self.msg)

    def post(self, request):
        print("这是一个post请求")
        print(request)

        return Response(data=self.msg)
