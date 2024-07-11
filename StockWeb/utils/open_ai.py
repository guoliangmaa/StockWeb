from openai import OpenAI

client = OpenAI(
    api_key="sk-tWxfaXXjo1NKvGjKvDamK3deP7kXDKNzbLqYDHn5aTGBFSYh",
    base_url="https://api.chatanywhere.tech/v1"
)


def gpt_35_api_stream(messages: list):
    stream = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=messages,
        stream=True,
    )
    text = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            # print(chunk.choices[0].delta.content, end="")
            text += chunk.choices[0].delta.content

    return text


# 授权成功
# 您的免费API Key为: sk-tWxfaXXjo1NKvGjKvDamK3deP7kXDKNzbLqYDHn5aTGBFSYh
# 请妥善保管，不要泄露给他人，如泄漏造成滥用可能会导致Key被封禁

if __name__ == '__main__':
    messages = [{'role': 'user', 'content': '帮我分析一下股票000001 直接分析 不用写问候消息 告诉我确定的信息 不确定的直接省略 不要分点 200字左右'}, ]
    print(gpt_35_api_stream(messages))
