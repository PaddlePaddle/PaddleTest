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

def compare(result, str, standard):
    result_value = ""
    standard_value = ""
    with open(result, "r", encoding="utf-8") as f:
        for line in f:
            if str in line:
                index = line[line.find(str) :]
                result_value += index

    with open(standard, "r", encoding="utf-8") as f:
        for line in f:
            standard_value += line
    assert result_value == standard_value


if __name__ == "__main__":
    obj = compare("test_PPDC1.log", "ClasOutput INFO", "PPDC1_standard.txt")
