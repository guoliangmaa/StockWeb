import asyncio

import edge_tts

# TEXT = "Hello World!"
TEXT = """
请输入正确的股票代码，例如：000001
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