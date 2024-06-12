from datetime import datetime

from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView

from StockWeb.utils.factory import get_mysql_engine


# 此控制器返回推荐的股票数据
class RecommendStockView(APIView):
    renderer_classes = [JSONRenderer]
    msg = {"msg": "success",
           "d": [],
           "predict": 0.0}
    retrain = True

    api_get_stock = "/api/stock/getData"

    def get(self, request):
        print(f"Get method, path = {request.get_full_path()}")
        engin = get_mysql_engine(database="predict_stock")
        sql = f'select * from recommend where date={datetime.today().strftime("%Y%m%d")}'
        with engin.connect() as connection:
            connection.execute(sql)
        return Response(data=self.msg)

    def post(self, request):
        print(f"post method, path = {request.get_full_path()}")
        return Response(data=self.msg)

    def get_data(self, request) -> Response:
        stock_code = request.GET.get("stock_code", "000001")  # 股票代码
        model_name = request.GET.get("model", "lstm").lower()  # 模型选择 默认都小写
        length = request.GET.get("length", 30 * 2)  # 看多少天以前的数据

        if model_name == "lstm":
            pass

        return Response(data=self.msg)
