import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.utils import formataddr

# 使用示例


from_email = "18830543657@163.com"  # 发件人网易云邮箱地址
from_password = "DDIIWJSPAMFVFABJ"  # 发件人邮箱的密码或授权码
# attachment_path = ["../1.txt", "../2.txt"]  # 如果没有附件，可以设为 None


def send_email(subject, body="这是邮件正文", to_email="772399887@qq.com", attachment_paths=None):
    # 创建 MIME 对象
    msg = MIMEMultipart()
    msg['From'] = formataddr(('Sender Name', from_email))
    msg['To'] = to_email
    msg['Subject'] = subject

    # 添加邮件正文
    msg.attach(MIMEText(body, 'plain'))

    # 添加附件（如果有）
    if attachment_paths:
        for attachment_path in attachment_paths:
            if os.path.isfile(attachment_path):
                with open(attachment_path, 'rb') as file:
                    part = MIMEApplication(file.read(), Name=os.path.basename(attachment_path))
                part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
                msg.attach(part)
            else:
                print(f"文件路径无效: {attachment_path}")

    # 连接到 SMTP 服务器并发送邮件
    try:
        with smtplib.SMTP_SSL('smtp.163.com', 465) as server:
            server.login(from_email, from_password)
            server.sendmail(from_email, to_email, msg.as_string())
            print("邮件发送成功!")
    except Exception as e:
        print(f"邮件发送失败: {e}")


send_email(subject="test")
