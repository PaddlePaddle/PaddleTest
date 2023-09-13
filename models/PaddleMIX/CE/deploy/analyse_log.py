import re

log_name = ''

# 打开日志文件并逐行读取内容
with open(log_name, 'r') as file:
    test_name = ''
    latency_data = []
    test_name_arr = []
    line = file.readline()

    while line:
        # 使用正则表达式提取测试名称
        test_match = re.match(r'==> (.*?)\n', line)
        if test_match:
            # 如果找到了测试名称，保存它
            test_name = test_match.group(1)
            test_name_arr.append(test_name)
        else:
            # 否则，尝试提取延迟数据
            latency_match = re.match(r'Mean latency: (.*?) s, p50 latency: (.*?) s, p90 latency: (.*?) s, p95 latency: (.*?) s.', line)
            if latency_match:
                # 如果找到了延迟数据，保存它
                latency_data.append(latency_match.groups())

        # 读取下一行
        line = file.readline()

# 输出提取的内容
for i in range(len(latency_data)):
    print(test_name_arr[i])
    print(f"Mean latency: {latency_data[i][0]} s, \
          p50 latency: {latency_data[i][1]} s, \
          p90 latency: {latency_data[i][2]} s, \
          p95 latency: {latency_data[i][3]} s.")
    print()
