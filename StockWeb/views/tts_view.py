import asyncio
from typing import BinaryIO

import edge_tts
from django.http import FileResponse
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView


class TTSView(APIView):
    renderer_classes = [JSONRenderer]
    # TEXT = "Hello World!"
    TEXT = """
    是的，你可以在文件写入操作完成后，直接在同一个 with 语句块中检查文件的大小。
    这可以确保你在打印文件对象时，所有数据已经成功写入。下面是一个更新的示例，演示如何在写入操作完成后检查文件的大小
    """
    VOICE = "zh-CN-YunxiaNeural"
    OUTPUT_FILE = "media/test.mp3"
    msg = {}

    async def get_voice(self) -> None:
        """Main function"""
        communicate = edge_tts.Communicate(self.TEXT, self.VOICE)
        with open(self.OUTPUT_FILE, "wb") as file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    print(f"WordBoundary: {chunk}")
            # 获取文件大小
            file_size = file.tell()
            print(f"File size: {file_size} bytes")

    def post(self, request) -> Response:
        # self.stock_code = request.GET.get("stock_code", "000001")
        self.TEXT = request.POST.get("text",
                                     "这可以确保你在打印文件对象时，所有数据已经成功写入。下面是一个更新的示例，演示如何在写入操作完成后检查文件的大小")
        print(f"post method, path = {request.get_full_path()}")
        asyncio.run(self.get_voice())
        self.msg["url"] = self.OUTPUT_FILE
        response = Response(data=self.msg)
        response['Access-Control-Allow-Origin'] = "*"
        return response
        # return FileResponse(open(self.OUTPUT_FILE, "rb"), as_attachment=True, filename="test.mp3")
