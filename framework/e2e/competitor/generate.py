#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
generator pytest cases
"""


import sys
from yaml_executor import RunCase

filedir = "../yaml/%s.yml" % sys.argv[1]
obj = RunCase(filedir)
all_cases = obj.get_all_case_name()

with open("test_{}.py".format(sys.argv[1]), "a") as f:
    f.write(
        (
            "#!/bin/env python\n"
            "# -*- coding: utf-8 -*-\n"
            "# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python\n"
            '"""\n'
            "test competitor cases\n"
            '"""\n'
            "\n"
            "import os\n"
            "import sys\n"
            "import pytest\n"
            "from yaml_executor import RunCase\n"
            "\n"
            "\n"
            'filedir = "../yaml/%s.yml" % sys.argv[1]\n'
            "obj = RunCase(filedir)\n"
            "\n"
            "\n"
        )
    )


for case in all_cases:
    desc = obj.get_docstring(case)
    with open("test_{}.py".format(sys.argv[1]), "a") as f:
        f.write(
            ("def test_{}():\n" '   """\n' "   {}\n" '   """\n' '   obj.run_case("{}")\n' "\n" "\n").format(
                case, desc, case
            )
        )

with open("test_{}.py".format(sys.argv[1]), "a") as f:
    f.write(
        ('if __name__ == "__main__":\n' '    pytest.main(["-sv", "--alluredir=../report/api", sys.argv[0]])\n' "\n")
    )
