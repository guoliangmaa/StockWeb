from openai import OpenAI

client = OpenAI(
    api_key="sk-tWxfaXXjo1NKvGjKvDamK3deP7kXDKNzbLqYDHn5aTGBFSYh",
    base_url="https://api.chatanywhere.tech/v1"
)


def gpt_35_api_stream(messages: list):
    stream = client.chat.completions.create(
        # model='gpt-3.5-turbo',
        model='gpt-4o-mini',
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
    messages = [{'role': 'user',
                 'content': '我这有一支股票，股票代码为000003，过去十天的收盘价分别为：8.75，8.73，8.73，8.57，8.34，8.49，8.28，8.44，8.45，8.59，预测的未来三天的收盘价为9.03，9.02，9.01。根据我给出的这些信息，请给出我这支股票的相关信息（对应的公司介绍）以及投资建议。相关信息少一点介绍，300字左右，不要分点'}, ]
    print(gpt_35_api_stream(messages))
