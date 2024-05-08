import json
import os
def get_html(json_data,html_data):
    """
    将测试结果以html格式写入文件
    """
    with open(json_data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    html_content = '<!DOCTYPE html>\n<html lang="zh">\n<head>\n<meta charset="UTF-8">\n<title>测试结果</title>\n</head>\n<body>\n'
    AGILE_JOB_BUILD_ID = os.getenv("AGILE_JOB_BUILD_ID")
    html_content += f'<h2>报告说明</h2>\
                    <p>此邮件为Modulus天级别监控邮件，目前进行动态图、动转静、动转静+组合算子、动转静+组合算子+编译器监控，监控粒度为L2和L4，由于模型运行时间较久，暂未监控L6级别。</p>\
                    <ul>\
                        <li>本次例行模型运行pytest测试报告见：<a href="https://paddle-qa.bj.bcebos.com/Modulus/{AGILE_JOB_BUILD_ID}/report.tar.gz">下载链接</a></li>\
                        <li>本次例行模型的具体测试日志见：<a href="https://paddle-qa.bj.bcebos.com/Modulus/{AGILE_JOB_BUILD_ID}/logs.tar.gz">下载链接</a></li>\
                    </ul>\
                    <h2>精度对齐数据列表</h2>\
                    <p>*精度对齐列表中红色表示未通过，其中N/A表示功能测试未通过，没有精度数据</p>\
                    <p>QA说明：此报告为天级别例行，有问题请联系suijiaxin</p>'
    html_content += '<table border="1">\n<tr><th></th><th>动态图</th><th></th><th>动转静</th><th></th><th>动转静+组合算子</th><th></th><th>动转静+组合算子+编译器</th><th></th></tr>\n'
    html_content += '<tr><td> </td><td>L2</td><td>L4</td><td>L2</td><td>L4</td><td>L2</td><td>L4</td><td>L2</td><td>L4</td></tr>\n'
    for model_name in data.keys():
        html_content += '<tr><td>' + model_name + '</td>'
        for sub_key in data[model_name].keys():
            for sub_sub_key in data[model_name][sub_key].keys():
                if 'atol' not in data[model_name][sub_key][sub_sub_key] or 'rtol' not in data[model_name][sub_key][sub_sub_key]:
                    html_content += '<td style="background-color:red;">N/A</td>'
                    continue
                atol = data[model_name][sub_key][sub_sub_key]['atol']
                rtol = data[model_name][sub_key][sub_sub_key]['rtol']
                if data[model_name][sub_key][sub_sub_key]['status'] == 'Failed':
                    html_content += '<td style="background-color:red;">' + 'atol: ' + str("{:.3e}".format(atol)) + ' \n ' + 'rotl:' + str("{:.3e}".format(rtol)) + '</td>'
                elif data[model_name][sub_key][sub_sub_key]['status'] == 'Success':
                    html_content += '<td style="background-color:green;">' + 'atol: ' + str("{:.3e}".format(atol)) + ' \n ' + 'rotl:' + str("{:.3e}".format(rtol)) + '</td>'
        html_content += '</tr>\n'

    html_content += '</table>'

    with open(html_data, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("HTML 表格已生成！")

if __name__ == '__main__':
    json_data = '../test_data.json'
    html_data = './index.html'
    get_html(json_data, html_data)
