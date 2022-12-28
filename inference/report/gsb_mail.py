"""
send mail
"""

# coding=utf-8
import os
import sys
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import datetime
import time

import mail_conf
import get_gsb


def mail(sender_addr, receiver_addr, subject, content, proxy):
    """
    send mail
    """
    msg = MIMEText(content, "html", "UTF-8")
    msg["From"] = sender_addr
    msg["To"] = receiver_addr
    msg["Subject"] = Header(subject, "UTF-8")

    server = smtplib.SMTP()
    server.connect(proxy)
    try:
        server.sendmail(sender_addr, msg["To"].split(","), msg.as_string())
        print("email send")
    except Exception as e:
        print("发送邮件失败:%s" % (e))
    finally:
        server.quit()


def create_gsb_table(gsb):
    """
    create gsb tables
    """
    subject = "[预测Benchmark][gsb] {}".format(gsb["task_dt"])
    content = """
        <html>
        <body>
        <div style="text-align:center;">
        </div>
    """

    for item in ["gpu", "cpu", "slim"]:
        # table1 GPU
        content += """
            <hr>
            <h2>{}</h2>
        """.format(
            item
        )

        if len(gsb[item]["value"]) < 1:
            continue
        if len(gsb[item]["frame"]) == 1 and gsb[item]["frame"][0] == "paddle":
            continue
        content += """
            环境信息：<br>{}<br><br>
            <table border="1" align=center>
        """.format(
            gsb[item]["env"]
        )

        content += """<tr>"""
        for table_title in gsb[item]["table_title"]:
            content += """<td>{}</td>""".format(table_title)
        content += """</tr>"""
        for mode in gsb[item]["value"].keys():
            for precision in gsb[item]["value"][mode].keys():
                for bs in gsb[item]["value"][mode][precision].keys():
                    content += """
                        <tr>
                        <td>{}</td>
                        <td>{}</td>
                        <td>{}</td>
                    """.format(
                        mode, precision, bs
                    )
                    for metric in gsb[item]["metric"]:
                        for frame in gsb[item]["frame"]:
                            if frame == "paddle":
                                continue
                            gsb_value = "-"
                            try:
                                if "gsb" in gsb[item]["value"][mode][precision][bs][frame][metric].keys():
                                    gsb_value = gsb[item]["value"][mode][precision][bs][frame][metric]["gsb"]
                            except:
                                pass
                            content += """
                                <td>{}</td>
                            """.format(
                                gsb_value
                            )
                    content += """</tr>"""
        content += """
            </table>
            <br><br>
        """

    return subject, content


def report_mail(gsb):
    """
    mail report
    """
    sender = mail_conf.SENDER
    reciver = mail_conf.RECIVER
    proxy = mail_conf.PROXY
    subject, content = create_gsb_table(gsb)
    mail(sender, reciver, subject, content, proxy)


if __name__ == "__main__":
    task_dt = sys.argv[1]
    gsb = get_gsb.get_gsb(task_dt)
    report_mail(gsb)
