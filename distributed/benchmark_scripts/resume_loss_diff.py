import json
from pathlib import Path
from datetime import datetime
import smtplib  
from email.mime.multipart import MIMEMultipart  
from email.mime.text import MIMEText  

RESUME_HEADER = ["task_name", "first_loss", "resume_loss", "diff_loss", \
                    "error_code_0", "result_zh_0", "result_en_0", \
                    "error_code_1", "result_zh_1", "result_en_1", \
                    "base_acc", "quant_acc", "diff_acc", \
                    "base_ppl", "quant_ppl", "diff_ppl"]

def get_result_list(logs_dir: str):
    data_list = []
    for p in Path(logs_dir).glob("**/*_N1C8_log"):
        # 过滤纯DP/PP的日志
        if "_resume" not in str(p) and "_predict" not in str(p) and "_quant" not in str(p):
            # ${model_item}_bs${base_batch_size}_${fp_item}_${run_mode}
            task_name = p.name.strip("llm-pdc_").strip("_log")
            try:
                first_loss, resume_loss, diff_loss = None, None, None
                error_code_0, result_zh_0, result_en_0 = None, None, None
                error_code_1, result_zh_1, result_en_1 = None, None, None
                base_acc, quant_acc, diff_acc, base_ppl, quant_ppl, diff_ppl = None, None, None, None, None, None
                if not p.exists():
                    print("pretrain未执行，请检查一下代码:", task_name)
                    break

                # resume_loss_diff
                resume_p = Path(str(p).replace("_N1C8_log", "_resume_N1C8_log")) 
                if not resume_p.exists():
                    print("resume日志不存在，请检查一下代码:", task_name)
                else:
                    for each in p.read_text().splitlines():
                        if "global_step: 51," in each:
                            first_loss = float(each.split("- loss: ")[1].split(", ")[0])
                            break
                                                
                    for each in resume_p.read_text().splitlines():
                        if "global_step: 51," in each:
                            resume_loss = float(each.split("- loss: ")[1].split(", ")[0])
                            break

                    if first_loss is None or resume_loss is None:
                        print(task_name, "日志未匹配到指定loss信息，请检查一下代码")
                    else:
                        diff_loss = abs(first_loss - resume_loss) * 10000
                
                # predict_cudaid0_result
                predict_p1 = Path(str(p).replace("_N1C8_log", "_predict_cudaid0_N1C8_log"))
                if predict_p1.exists():
                    predict_p = predict_p1
                else:
                    predict_p = Path(str(p).replace("_N1C8_log", "_predict_N1C8_log"))
                if "pretrain" in str(predict_p):
                    error_code_0, result_zh_0, result_en_0 = "-", "-", "-"
                elif not predict_p.exists():
                    print("predict日志不存在，请检查一下代码:", predict_p)
                else:
                    for each in predict_p.read_text().splitlines():
                        if "error_code: " in each:
                            error_code_0 = int(each.split("error_code: ")[1].split(", ")[0])
                        if "result_zh: " in each:
                            result_zh_0 = each.split("result_zh: ")[1]
                        if "result_en: " in each:
                            result_en_0 = each.split("result_en: ")[1]

                # predict_cudaid1_result
                predict_p2 = Path(str(p).replace("_N1C8_log", "_predict_cudaid1_N1C8_log"))
                if predict_p2.exists():
                    predict_p = predict_p2
                else:
                    predict_p = Path(str(p).replace("_N1C8_log", "_predict_N1C8_log"))
                if "pretrain" in str(predict_p):
                    error_code_1, result_zh_1, result_en_1 = "-", "-", "-"
                elif not predict_p.exists():
                    print("predict日志不存在，请检查一下代码:", predict_p)
                else:
                    for each in predict_p.read_text().splitlines():
                        if "error_code: " in each:
                            error_code_1 = int(each.split("error_code: ")[1].split(", ")[0])
                        if "result_zh: " in each:
                            result_zh_1 = each.split("result_zh: ")[1]
                        if "result_en: " in each:
                            result_en_1 = each.split("result_en: ")[1]

                # quant_result
                quant_p = Path(str(p).replace("_N1C8_log", "_quant_N1C8_log"))
                if "pretrain" in str(quant_p):
                    base_acc, quant_acc, diff_acc, base_ppl, quant_ppl, diff_ppl = "-", "-", "-", "-", "-", "-"
                elif not quant_p.exists():
                    print("quant_p日志不存在，请检查一下代码:", quant_p)
                else:
                    for each in p.read_text().splitlines():
                        if "eval_accuracy: " in each:
                            base_acc = float(each.split("eval_accuracy: ")[1].split(",")[0])
                        if "eval_ppl: " in each:
                            base_ppl = float(each.split("eval_ppl: ")[1].split(",")[0])
                                                
                    for each in quant_p.read_text().splitlines():
                        if "eval_accuracy: " in each:
                            quant_acc = float(each.split("eval_accuracy: ")[1].split(",")[0])
                        if "eval_ppl: " in each:
                            quant_ppl = float(each.split("eval_ppl: ")[1].split("\x1b[0m")[0])

                    if base_acc is None or quant_acc is None:
                        print(task_name, "日志未匹配到指定eval_accuracy信息，请检查一下代码")
                    else:
                        diff_acc = str(round((quant_acc - base_acc)/base_acc * 100, 3)) + "%"

                    if base_ppl is None or quant_ppl is None:
                        print(task_name, "日志未匹配到指定eval_ppl信息，请检查一下代码")
                    else:
                        diff_ppl = str(round((quant_ppl - base_ppl)/base_ppl * 100, 3)) + "%"
                
                data = [task_name, first_loss, resume_loss, diff_loss, \
                        error_code_0, result_zh_0, result_en_0, \
                        error_code_1, result_zh_1, result_en_1, \
                        base_acc, quant_acc, diff_acc, \
                        base_ppl, quant_ppl, diff_ppl]
                data_list.append(data)
            except Exception as e:
                print(e)
                print(f"task_name {task_name} 可能有错误，请检查一下！！！！")
    return data_list


def construct_email_content(data_list):
    alarm_html = '<head><style> .center {margin-left: auto;margin-right: auto;}</style></head> \
                    <table border="1" cellpadding="5" class="center"><tr>'
    for item in RESUME_HEADER:
        alarm_html += '<td>%s</td>' % item
    alarm_html += '</tr>'

    for row in data_list:
        alarm_html += '<tr>'
        for i in range(len(row)):
            if i == 3 and 0.0 != row[i]:
                alarm_html += '<td  style="background-color:red">%s</td>' % row[i]
            elif i == 4 and 0 != row[i] and "-" != row[i]:
                alarm_html += '<td  style="background-color:red">%s</td>' % row[i]
            elif i == 7 and 0 != row[i] and "-" != row[i]:
                alarm_html += '<td  style="background-color:red">%s</td>' % row[i]
            elif i == 12 and "0.0%" != row[i] and "-" != row[i]:
                alarm_html += '<td  style="background-color:red">%s</td>' % row[i]
            elif i == 15 and "0.0%" != row[i] and "-" != row[i]:
                alarm_html += '<td  style="background-color:red">%s</td>' % row[i]
            else:
                alarm_html += '<td>%s</td>' % row[i]
        alarm_html += "</tr>"
    alarm_html += "</table>"    
    return alarm_html

  

def send_email_resume(dir_name):
    with open(dir_name + "/temp_info.json", "r") as handler:
        temp_data = json.load(handler)
        ENV_INFO = temp_data["ENV_INFO"]
    tmp_dt = datetime.strptime(ENV_INFO["frame_commit_dt"], "%Y-%m-%d %H:%M:%S")
    ENV_INFO["frame_commit_dt"] = tmp_dt

    data_list = get_result_list(ENV_INFO["dir_name"] + "/dynamic/train_log/")
    html_results = construct_email_content(data_list)
    
    # 创建邮件  
    msg = MIMEMultipart()  
    msg['From'] = "paddle_benchmark@baidu.com" 
    msg['To'] =  ','.join([ENV_INFO["email_address"]]) 
    msg['Subject'] = "【分布式PDC大模型训练Benchmark_PDC_{}_{}_CUDA{}_Python{}_{}】运行结果报警，请查看".format(
        ENV_INFO["frame_commit_dt"].strftime("%Y-%m-%d"), 
        ENV_INFO["device_type"], 
        ENV_INFO["cuda_version"], 
        ENV_INFO["python_version"], 
        ENV_INFO["frame_branch"])  
    
    # 将CSV内容添加到邮件正文
    msg.attach(MIMEText(html_results, _subtype="html", _charset="UTF-8"))

    server = smtplib.SMTP()
    server.connect("proxy-in.baidu.com") 
    try:
        server.sendmail("paddle_benchmark@baidu.com", msg['To'].split(','), msg.as_string())
        print("resume_email send")
    except Exception as e:
        print("发送邮件失败:%s" % (e))
    finally:
        server.quit()



if __name__ == "__main__": 
    # data_list = get_result_list("./dynamic/train_log/")
    # html_results = construct_email_content(data_list)
    send_email_resume(dir_name)
    