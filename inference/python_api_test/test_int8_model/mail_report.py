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


def add_table_gsb(jingping, gsb, metric_list):
    """
    gsb
    """
    content = """
        <br><br>
        <table border="1" align=center>
        <caption bgcolor="#989898">与{}比的GSB</caption>
    """.format(
        jingping
    )
    # line1
    D_gsb = 3
    content += "<tr><td></td>"
    for k in metric_list:
        content += "<td>{}</td>".format(k)
        for i in range(D_gsb - 1):
            content += "<td></td>"
    content += "</tr>"
    # line2
    content += "<tr><td></td>"
    for k in metric_list:
        content += "<td>GSB</td>"
        content += "<td>下降数(占比)</td>"
        content += "<td>上升数(占比)</td>"
    content += "</tr>"
    # line data
    for mode, info in gsb.items():
        content += "<tr>"
        content += "<td>{}</tr>".format(mode)
        for item in metric_list:
            _gsb = info[item]["gsb"]
            _b = "{} ({})".format(info[item]["b"], round(info[item]["b_ratio"], 3))
            _g = "{} ({})".format(info[item]["g"], round(info[item]["g_ratio"], 3))
            content += "<td>{}</tr>".format(_gsb)
            content += "<td>{}</tr>".format(_b)
            content += "<td>{}</tr>".format(_g)
        content += "</tr>"
    content += """
        </table>
        <br><br>
    """
    return content


def create_table_day(task_dt, env, gsb, detail, mode_list, metric_list, jingping_list):
    """
    create subject and content of mail
    """
    subject = "[预测-量化Benchmark][{}]{}执行结果".format(env["device"], task_dt)

    content = """
        <html>
        <body>
        <div style="text-align:center;">
        </div>
    """

    # env
    content += """
        docker_image:{}
        <br>
        frame:{}
        <br>
        frame_branch:{}
        <br>
        frame_commit:{}
        <br>
        阈值:{}
        <br>
    """.format(
        env["docker_image"],
        env["frame"],
        env["frame_branch"],
        env["frame_commit"],
        env["threshold"],
    )

    # table1 gsb
    for jingping in jingping_list:
        content += add_table_gsb(jingping, gsb[jingping], metric_list)

    # table3 detail
    D = 1 + 2 * len(jingping_list)
    content += """
        <table border="1" align=center>
        <caption bgcolor="#989898">详细数据</caption>
    """
    # line1
    content += "<tr><td></td>"
    n = len(metric_list) * D
    for m in mode_list:
        content += "<td>{}</td>".format(m)
        for i in range(n - 1):
            content += "<td></td>"
    content += "</tr>"
    # line2
    content += "<tr><td></td>"
    for m in mode_list:
        for k in metric_list:
            content += "<td>{}</td>".format(k)
            for i in range(D - 1):
                content += "<td></td>"
    content += "</tr>"
    # line3
    content += "<tr><td></td>"
    n = len(metric_list) * len(mode_list)
    for i in range(n):
        content += "<td>实际值</td>"
        for jingping in jingping_list:
            content += "<td>{}值</td>".format(jingping)
            content += "<td>diff-{}</td>".format(jingping)
    content += "</tr>"
    # line date
    models = list(detail["base"].keys())
    for model in models:
        content += "<tr>"
        content += "<td>{}</td>".format(model)
        for m in mode_list:
            for k in metric_list:
                content += "<td>{}</td>".format(round(detail["base"][model][m][k]["benchmark"], 3))
                for jingping in jingping_list:
                    if (model in detail[jingping].keys()) and (m in detail[jingping][model].keys()):
                        content += "<td>{}</td>".format(round(detail[jingping][model][m][k]["base"], 3))
                        content += "<td>{}</td>".format(round(detail[jingping][model][m][k]["diff"], 3))
                    else:
                        content += "<td>-</td>"
                        content += "<td>-</td>"
        content += "</tr>"

    content += """
        </table>
        </body>
        </html>
    """

    return subject, content


def report_day(task_dt, env, gsb, detail, mode_list, metric_list, jingping_list):
    """
    天级报告
    """
    sender = mail_conf.SENDER
    reciver = mail_conf.RECIVER
    proxy = mail_conf.PROXY
    subject, content = create_table_day(task_dt, env, gsb, detail, mode_list, metric_list, jingping_list)
    mail(sender, reciver, subject, content, proxy)


if __name__ == "__main__":
    pass
