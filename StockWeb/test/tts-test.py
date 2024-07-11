import asyncio

import edge_tts

# TEXT = "Hello World!"
TEXT = """
是的，你可以在文件写入操作完成后，直接在同一个 with 语句块中检查文件的大小。
这可以确保你在打印文件对象时，所有数据已经成功写入。下面是一个更新的示例，演示如何在写入操作完成后检查文件的大小
"""
VOICE = "zh-CN-YunxiaNeural"
OUTPUT_FILE = "test.mp3"


async def amain() -> None:
    """Main function"""
    communicate = edge_tts.Communicate(TEXT, VOICE)
    with open(OUTPUT_FILE, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                print(f"WordBoundary: {chunk}")
        # 获取文件大小
        file_size = file.tell()
        print(f"File size: {file_size} bytes")


if __name__ == "__main__":
    asyncio.run(amain())