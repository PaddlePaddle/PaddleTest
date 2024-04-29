import json

def get_html(json_data,html_data):
    """
    将测试结果以html格式写入文件
    """
    with open(json_data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    html_content = '<!DOCTYPE html>\n<html lang="zh">\n<head>\n<meta charset="UTF-8">\n<title>测试结果</title>\n</head>\n<body>\n'
    html_content += '<table border="1">\n<tr><th></th><th>动态图</th><th></th><th>动转静</th><th></th><th>动转静+组合算子</th><th></th></tr>\n'
    html_content += '<tr><td> </td><td>L2</td><td>L4</td><td>L2</td><td>L4</td><td>L2</td><td>L4</td></tr>\n'
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
    json_data = './test_data.json'
    html_data = './html_result.html'
    get_html(json_data, html_data)
