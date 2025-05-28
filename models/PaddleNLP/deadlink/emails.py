"""
auto send email
@auther liuhuanling
"""
# -*- coding:utf8 -*-
import csv
import smtplib
import mimetypes
from email import encoders
from smtplib import SMTP_SSL
from email.mime.text import MIMEText
from email.header import Header
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from PIL import Image
from io import BytesIO, StringIO
import requests
import time
import threading


SUFFIX_TO_FORMAT = {
    'jpg': 'JPEG',
    'jpeg': 'JPEG',
    'png': 'PNG'
}


def get_image_format_by_suffix(image_url):
    """
    根据图片后缀获取图片格式
    :param image_url:
    :return:
    """
    if not isinstance(image_url, (str, unicode)):
        return 'JPEG'
    suffix = image_url.split('.')[-1]
    return SUFFIX_TO_FORMAT.get(str(suffix), 'JPEG')


class Email(object):
    """
    define email class
    """
    _from = 'auto_send@baidu.com'
  
    def send_an_email(self, to, content):
        """
        params to: email recevier
        params content: email content
        create connection and send email
        """
        smtp = smtplib.SMTP()
        smtp.connect("10.169.0.216")
        to = to.split(",")
        smtp.sendmail(self._from, to, content.as_string())

    def base_send_email(self, to, subject, content):
        """
        params to: email recevier
        params subject: email subject
        params content: email content
        process email content
        """
        message = MIMEMultipart('related')
        message['Subject'] = Header(subject, 'utf8')
        message['From'] = self._from
        message['To'] = to

        mime_text = MIMEText(content, 'html', 'utf-8')
        message.attach(mime_text)
        return message

    def get_mime_images(self, image_urls):
        """
        image_urls: 图片列表
        下载图片的内容
        """
        mime_images = []
        for url in image_urls:
            response = requests.get(url)
            buff = BytesIO(response.content)
            image_format = get_image_format_by_suffix(url)
            mime_image = MIMEImage(buff.read(), _subtype=image_format)
            buff.close()
            mime_images.append(mime_image)
        return mime_images

    def send_email_with_image(self, to, subject, content, file_path=None, image_urls=None, **kwargs):
        """
        发送带附件图片的邮件
        :param to:
        :param subject:
        :param content:
        :param image_urls: 图片url列表
        :return:
        """
        file_attach = kwargs.get("file_attach", list())

        message = self.base_send_email(to, subject, content)
        mime_images = self.get_mime_images(image_urls) if image_urls else list()
        for item in mime_images:
            item.add_header('Content-ID', '')
            message.attach(item)
        
        if file_attach:
            for item in file_attach:

                file_name = item.split("/")[-1]
                ctype, encoding = mimetypes.guess_type(item)
                if ctype is None or encoding is not None:
                    ctype = "application/octet-stream"
                maintype, subtype = ctype.split("/", 1)
                fp = open(item, "rb")
                attachment = MIMEBase(maintype, subtype)
                attachment.set_payload(fp.read())
                fp.close()
                encoders.encode_base64(attachment)
                attachment.add_header("Content-Disposition", "attachment", filename=file_name)
                message.attach(attachment)

        self.send_an_email(to, message)

    def send_email_for_threads(self, email_list):
        """
        多线程发送邮件
        :param email_list:
        :return:
        """
        thread_list = []
        for params in email_list:
            thread = threading.Thread(target=self.send_email_with_image,
                                      kwargs=params)
            thread_list.append(thread)

        for thread in thread_list:
            thread.start()

        for thread in thread_list:
            thread.join()


def send_email(to_email, subject, content="", file_path=None, images=None, file_attach=list()):
    """
    params: to_email 收件人地址
    params subject: 邮件主题
    params content: 邮件内容
    params file_path: 显示的内容，只支持一个
    params images: 邮件带图片，有多张则以list传递
    params file_attach: 附件带文件
    """
    if content:
        email_content = content
    elif file_path:
        # 现将结果处理成html，再读取
        if file_path.endswith("csv"):
            with open(file_path, "r") as f:
                f_csv = csv.reader(f)
                cols = next(f_csv)
                return_str = '<table border="1" style="border-collapse: collapse; border-spacing: 7px;"><tr style="background-color:green;">'
                for key in cols:
                    return_str = return_str + '<th class="header">' + str(key) + '</th>'
                return_str = return_str + '</tr>'
                for key in f_csv:
                    return_str = return_str + '<tr>'
                    for item in key:
                        return_str = return_str + '<td>' + str(item) + '</td>'
                email_content = return_str + '</tr></table>'
        else:
            with open(file_path, "r", encoding='utf-8') as f:
                email_content= f.read()
    else:
        email_content = subject + "详情查看附件"
    email_ins = Email()
    email_ins.send_email_with_image(to=to_email, subject=subject, content=email_content, image_urls=images, file_attach=file_attach)
    
if __name__ == "__main__":
    send_email("liuhuanling@baidu.com", "Test email", file_path="./default_template.html")

