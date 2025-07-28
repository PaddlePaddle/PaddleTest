"""
解析死链结果并发邮件
@auther liuhuanling
"""
import io
import csv
import argparse
from emails import send_email
header = ['状态', 'url', '文件名', 'NLP相对路径', '分支']

def process_result(data_path, email_addr, email_sub):
    # 处理结果转换成csv文件，将404 以及超时的放在前面成功的放在后面
    # 处理之后发送邮件
    print(data_path)
    succeed_data = list()
    failed_data = list()
    other_data = list()
    with open(data_path, "r", encoding="utf-8") as readfile:
        for line in readfile.readlines():
            if line:
                newline = line.strip()
                newline = newline.strip("\n")
                content = newline.split("\t")
                if len(content) > 5:
                    content = [item for item in content if item]
                status = content[0]
                try:
                    status = int(status)
                except:
                    status = status
                if type(status) == int:
                    if status == 200 or (str(status).startswith("3")) or (str(status).startswith("5")) or status==403 :
                        # 2 开头或者3开头的算成功，其他算失败
                        succeed_data.append(content)
                    else:
                        failed_data.append(content)
                else:
                    other_data.append(content)
    file_path = None
    file_attach = []
    if succeed_data:
        with open("./成功.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, dialect='excel')
            writer.writerow(header)
            writer.writerows(succeed_data)
        file_attach.append("./成功.csv")
    if failed_data:
        file_path = "./失败.csv"
        with open("./失败.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, dialect='excel')
            writer.writerow(header)
            writer.writerows(failed_data)
        file_attach.append("./失败.csv")
    if other_data:
        with open("./超时或其他情况.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, dialect='excel')
            writer.writerow(header)
            writer.writerows(other_data)
        file_attach.append("./超时或其他情况.csv")

    
    send_email(email_addr, email_sub, file_path=file_path, file_attach=file_attach)

def parse_args():
    """
    接收和解析命令传入的参数
    """
    parser = argparse.ArgumentParser("Tool for running for dead line")
    parser.add_argument("--data_path", help="普通文件", type=str)
    parser.add_argument("--email_addr", help="收件人邮箱", type=str, default=None)
    parser.add_argument("--email_sub", help="邮件主题", type=str, default=None)
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    email_addr = args.email_addr
    email_sub = args.email_sub
    process_result(data_path, email_addr, email_sub)
