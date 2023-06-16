import yaml
import os
import subprocess

api_list = [
    "all_gather",
    "all_reduce",
    "alltoall",
    "alltoall_single",
    "broadcast",
    "send_recv",
    "reduce",
    "reduce_scatter",
    "scatter",
]
loops = 50


def get_average(file_loops, case):
    counters = {}
    averages = {}

    f = open(file_loops, encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        if "'" + case + "'" in line:
            for key, value in eval(line).items():
                for num, item in value.items():
                    if num not in counters:
                        counters[num] = {"time": 1, "algbw": 1}
                        averages[num] = {"time": item["time"], "algbw": item["algbw"]}
                    else:
                        counters[num]["time"] += 1
                        counters[num]["algbw"] += 1
                        averages[num]["time"] += item["time"]
                        averages[num]["algbw"] += item["algbw"]

    # 计算每个数字对应的平均值
    for num in counters.keys():
        averages[num]["time"] /= counters[num]["time"]
        averages[num]["algbw"] /= counters[num]["algbw"]

    # 关闭文件
    f.close()
    avg_res = {case: averages}

    # 写入文件，存档
    with open("log_avg", "a", encoding="utf8") as f:
        f.write(str(avg_res) + "\n")
        f.flush()

    os.system("rm -rf ./log")
    return avg_res


def compare(case, res_dict):
    f = open("./base_value", encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        if line.find("'" + case + "'") != -1:
            base_dict = eval(line.strip("\n"))

    diff_dict = {}
    for key, value in res_dict.items():
        for num, item in value.items():
            time_diff = (item["time"] - base_dict[key][num]["time"]) / base_dict[key][num]["time"] * 100
            algbw_diff = (item["algbw"] - base_dict[key][num]["algbw"]) / base_dict[key][num]["algbw"] * 100
            diff_dict[num] = {"time": time_diff, "algbw": algbw_diff}

    diff_res = {case: diff_dict}

    # 写入文件，存档
    with open("log_diff", "a", encoding="utf8") as f:
        f.write(str(diff_res) + "\n")
        f.flush()

    return diff_res


def gather_dict(case):
    all_dict = {}
    f_dict = {"base": "./base_value", "avg": "./log_avg", "diff": "./log_diff"}

    for key, value in f_dict.items():
        f = open(value, encoding="utf-8")
        lines = f.readlines()
        for line in lines:
            if line.find("'" + case + "'") != -1:
                all_dict[key] = eval(line.strip("\n"))[case]
                for num, item in all_dict[key].items():
                    all_dict[key][num]["time_" + key] = all_dict[key][num].pop("time")
                    all_dict[key][num]["algbw_" + key] = all_dict[key][num].pop("algbw")
    f.close()
    # print(all_dict)

    res_dict = {}
    for key, value in all_dict.items():
        for num, item in value.items():
            if num not in res_dict.keys():
                res_dict[num] = {
                    "time_avg": 0,
                    "time_base": 0,
                    "time_diff": 0,
                    "algbw_avg": 0,
                    "algbw_base": 0,
                    "algbw_diff": 0,
                }
            for key_item, value_item in item.items():
                res_dict[num][key_item] = value_item
    result = {case: res_dict}

    # 写入文件，存档
    with open("log_result", "a", encoding="utf8") as f:
        f.write(str(result) + "\n")
        f.flush()

    return result


def main():
    f = open("config.yaml", "rb")
    yaml_config = yaml.load(f, Loader=yaml.FullLoader)

    for key, value in yaml_config.items():
        if key in api_list:
            for case in value.keys():
                cmd = "python -m paddle.distributed.launch --devices=0,1,2,3,4,5,6,7 " + key + ".py --case_name " + case
                print(cmd)
                for i in range(loops):
                    pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    out, err = pro.communicate()
                    print(out)
                    pro.wait()
                    pro.returncode == 0
                # 求均值，写入文件log_avg
                avg_res = get_average("./log/workerlog.0", case)
                # 求diff，写入文件log_diff
                # diff = compare(case, avg_res)
                # 得出汇总结果，写入log_result
                # result = gather_dict(case)


if __name__ == "__main__":
    main()
    os.system("cat log_avg")