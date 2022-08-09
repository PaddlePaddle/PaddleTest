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
from jittrans import JitTrans

yaml_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "yaml", "nn.yml")
yml = YamlLoader(yaml_path)
cases = yml.get_all_case_name()
# print(cases)
base_case = []
api_list = []
for case in cases:
    if "_base" in case:
        base_case.append(case)
for base in base_case:
    tmp = base.replace("_base", "")
    api_list.append(tmp)
print(api_list)

# str_all = ""

for a in api_list:
    str_all = ""
    for case in cases:
        if a + "_" in case:
            tmp = (
                "def test_{}():\n"
                '    """test {}"""\n'
                "    jit_case = JitTrans(case=yml.get_case_info('{}'))\n"
                "    jit_case.jit_run()\n"
                "\n"
                "\n".format(case, case, case)
            )
            str_all += tmp

    with open("pytest_dir/test_{}.py".format(a), "w") as f:
        f.write(
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
            'yaml_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "yaml", "nn.yml")\n'
            "yml = YamlLoader(yaml_path)\n"
            "\n"
            "\n"
        )
        f.write(str_all)
