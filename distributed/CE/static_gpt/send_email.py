#!/usr/bin/env python3
# -*- coding:UTF-8 -*-


import os
import sys
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.header import Header


class mailmsg(object):
    def __init__(self, sender, receiver, subject, content, attach_files=None):
        self.sender = sender
        self.receiver = receiver
        self.subject = subject
        self.content = content
        self.attach_files = attach_files

    def send_email(self, attach_charset="UTF-8"):
        """
        send email api
        """
        # Parameter preprocessing
        if self.sender is None:
            print("sender is none")
        # Construct mail
        msg = MIMEMultipart("related")
        msg["Subject"] = Header(self.subject, "UTF-8")
        msg["From"] = "<%s>" % self.sender
        receiver_list = self.receiver.split(";")
        receiver_to_list = receiver_list[0].split(",")
        if len(receiver_list) > 1:
            receiver_cc_list = receiver_list[1].split(",")
        else:
            receiver_cc_list = []
        # msg["To"] = Header(";".join(receiver_to_list), "UTF-8")
        # msg["Cc"] = Header(";".join(receiver_cc_list), "UTF-8")
        msg["To"] = ";".join(receiver_to_list)
        msg["Cc"] = ";".join(receiver_cc_list)
        now_time = datetime.datetime.now()
        now_time_str = datetime.datetime.strftime(now_time, '%Y-%m-%d %H:%M:%S')

        # Add text
        html_text = '''
                <p></p>
                ''' + self.content + '''
                <p></p>
                <table border=1>
                <tr><th>case</th><th>baseline_hybrid</th><th>result_auto</th></tr>
                <tr><td>DP2</td><td>7.116779</td><td>'''+sys.argv[1]+'''</td></tr>
                <tr><td>MP2</td><td>None</td><td>'''+sys.argv[2]+'''</td></tr>
                <tr><td>DP2MP2</td><td>7.097414</td><td>'''+sys.argv[3]+'''</td></tr>
                </table>
                <p></p>
                
                <p>Task link：<a href=""></a></p>
                Machine address：gzns-ps-201608-m02-www028.gzns.baidu.com (Account password private chat)
                </p>
                '''

        msg.attach(MIMEText(html_text, _subtype="html", _charset="UTF-8"))
        # Add attachment
        if self.attach_files is not None:
            for file in self.attach_files:
                part = MIMEBase("application", "octet-stream")
                file_content = ""
                infile = open(file, "rb")
                try:
                    file_content = infile.read()
                finally:
                    infile.close()
                part.set_payload(file_content, attach_charset)
                part.add_header("Content-Disposition", "attachment; filename=%s" % os.path.basename(file))
                msg.attach(part)
        # Send mail
        smtp = smtplib.SMTP()
        # smtp.connect("hotswap-in.baidu.com")
        smtp.connect("proxy-in.baidu.com")
        try:
            print("start send")
            smtp.sendmail(self.sender, receiver_to_list + receiver_cc_list, msg.as_string())
        except Exception as e:
            print("Failed to send mail:%s" % (e))
        finally:
            # print("send finally")
            smtp.quit()


if __name__ == "__main__":
    # send_baidu_mail= mailmsg("liujie44@baidu.com", "liujie44@baidu.com", "Email Test", "my content")
    send_baidu_mail = mailmsg("liujie44@baidu.com", "liujie44@baidu.com", "gpt CE Report",
            "results:")
    send_baidu_mail.send_email()
