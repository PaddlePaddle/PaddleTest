#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test jit cases
"""
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "utils"))

from utils.yaml_loader import YamlLoader

# from jittrans import JitTrans

yaml_type = "base"
yaml_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "yaml", yaml_type + ".yml")
yml = YamlLoader(yaml_path)
cases = yml.get_all_case_name()
# print(cases)
base_case = []
all_case = []
api_list = []
for case in cases:
    if "_base" in case:
        base_case.append(case)
    all_case.append(case)
for base in base_case:
    tmp = base.replace("_base", "")
    api_list.append(tmp)
print(all_case)
print(api_list)
print(len(api_list))


# a = all_case.pop(0)
# print(all_case)
api_list_ = []
for a in api_list:
    api_list_.append(a + "_")
print(api_list_)
# api_dict = {}
# for c in all_case:

for case in all_case:
    tmp = (
        "def test_{}():\n"
        '    """test {}"""\n'
        "    jit_case = JitTrans(case=yml.get_case_info('{}'))\n"
        "    jit_case.jit_run()\n"
        "\n"
        "\n".format(case, case, case)
    )
    # str_all += tmp
    with open("pytest_dir/test_{}.py".format(case), "w") as f:
        f.write(
            (
                "#!/bin/env python\n"
                "# -*- coding: utf-8 -*-\n"
                "# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python\n"
                '"""\n'
                "test jit cases\n"
                '"""\n'
                "\n"
                "import os\n"
                "import sys\n"
                "\n"
                "sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))\n"
                'sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "utils"))\n'
                "\n"
                "from utils.yaml_loader import YamlLoader\n"
                "from jittrans import JitTrans\n"
                "\n"
                'yaml_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "yaml", "{}.yml")\n'
                "yml = YamlLoader(yaml_path)\n"
                "\n"
                "\n"
            ).format(yaml_type)
        )
        f.write(tmp)

# str_all = ""
#
# for a in api_list_:
#     str_all = ""
#     for case in all_case:
#         if a in case:
#             tmp = (
#                 "def test_{}():\n"
#                 '    """test {}"""\n'
#                 "    jit_case = JitTrans(case=yml.get_case_info('{}'))\n"
#                 "    jit_case.jit_run()\n"
#                 "\n"
#                 "\n".format(case, case, case)
#             )
#             str_all += tmp
#
#     with open("pytest_dir/test_{}.py".format(a), "w") as f:
#         f.write(
#             "#!/bin/env python\n"
#             "# -*- coding: utf-8 -*-\n"
#             "# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python\n"
#             '"""\n'
#             "test jit cases\n"
#             '"""\n'
#             "\n"
#             "import os\n"
#             "import sys\n"
#             "\n"
#             "sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))\n"
#             'sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "utils"))\n'
#             "\n"
#             "from utils.yaml_loader import YamlLoader\n"
#             "from jittrans import JitTrans\n"
#             "\n"
#             'yaml_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "yaml", "nn.yml")\n'
#             "yml = YamlLoader(yaml_path)\n"
#             "\n"
#             "\n"
#         )
#         f.write(str_all)
