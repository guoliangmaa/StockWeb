from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView
from sqlalchemy import text
from StockWeb.utils.factory import get_mysql_engine
from StockWeb.utils.database_stock import warning_stock


# 此控制器返回推荐的股票数据
class WarningStockView(APIView):
    renderer_classes = [JSONRenderer]
    msg = {"msg": "success",
           "d": [],
           "predict": 0.0}

    def get(self, request):
        print(f"Get method, path = {request.get_full_path()}")
        result = warning_stock()
        print(result)

        self.msg["data"] = result
        response = Response(data=self.msg)
        response['Access-Control-Allow-Origin'] = "*"
        return response

    def post(self, request):
        print(f"post method, path = {request.get_full_path()}")
        return Response(data=self.msg)
