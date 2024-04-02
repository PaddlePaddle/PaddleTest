# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
analyse_log
"""
import re
import argparse


def process_log_file(log_name):
    test_name = ''
    latency_data = []
    test_name_arr = []

    try:
        with open(log_name, 'r') as file:
            line = file.readline()

            while line:
                test_match = re.match(r'==> (.*?)\n', line)
                if test_match:
                    test_name = test_match.group(1)
                    test_name_arr.append(test_name)
                else:
                    latency_match = re.match(
                        r'Mean latency: (.*?) s, p50 latency: (.*?) s, p90 latency: (.*?) s, p95 latency: (.*?) s.', line)
                    if latency_match:
                        latency_data.append(latency_match.groups())
                line = file.readline()
    except FileNotFoundError:
        print(f"File '{log_name}' not found.")
        return

    res_log_name = log_name.replace('.log', '-res.log')

    with open(res_log_name, 'w', encoding='utf-8') as file:
        for i in range(len(latency_data)):
            file.write(test_name_arr[i] + '\n')
            file.write(f"Mean latency: {latency_data[i][0]} s, "
                       f"p50 latency: {latency_data[i][1]} s, "
                       f"p90 latency: {latency_data[i][2]} s, "
                       f"p95 latency: {latency_data[i][3]} s.\n")
            file.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_name', required=True,
                        help='Name of the log file')
    args = parser.parse_args()
    log_name = args.log_name
    process_log_file(log_name)
