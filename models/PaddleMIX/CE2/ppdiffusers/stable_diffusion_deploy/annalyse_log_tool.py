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
annalyse_log_tool
"""
import argparse


def check_log(file_path):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'FAILED' in line:
                    exit(1)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', required=True,
                        help='file_path of the log file')
    args = parser.parse_args()
    file_path = args.file_path
    check_log(file_path)
