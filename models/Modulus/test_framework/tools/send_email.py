from calendar import c
import smtplib
from email.mime.text import MIMEText
from email.header import Header
def email_send(sender, receivers, subject, content):
    """
    send
    """
    sender = sender
    message = MIMEText(content, 'html', 'utf-8')
    message["To"] = Header(','.join(receivers))

    message['Subject'] = Header(subject, 'utf-8')

    try:
        smtpObj = smtplib.SMTP('mail2-in.baidu.com')
        smtpObj.sendmail(sender, receivers, message.as_string())
        return True
    except smtplib.SMTPException as e:
        return e
def send_email(sender, receivers, subject):
    """
    send
    """
    with open('./html_result/index.html', 'r', encoding='utf-8') as f:
        content = f.read()
    email_send(sender, receivers, subject, content)
    print('send success')

if __name__ == '__main__':
    sender = 'auto_send@baidu.com'
    receivers = ['suijiaxin@baidu.com']
    subject = 'Modulus CE Test'
    send_email(sender, receivers, subject)