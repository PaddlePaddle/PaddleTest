#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
main test
"""
import os
import shutil
import platform


class ClasV2Test(object):
    """
    test Clas to onnx tipc
    """

    def __init__(self):
        if os.path.exists("tipc_models_url_PaddleClas_latest.txt"):
            os.remove("tipc_models_url_PaddleClas_latest.txt")

        self.txt_url = (
            "https://paddle-qa.bj.bcebos.com/fullchain_ce_test/"
            "model_download_link/tipc_models_url_PaddleClas_latest.txt"
        )

        os.system("wget -q --no-proxy {}".format(self.txt_url))

        self.model_url_list = []
        for line in open("tipc_models_url_PaddleClas_latest.txt"):
            self.model_url_list.append(line)

        self.opset_v_list = [10, 11, 12]

    def prepare_resource(self, tgz_url):
        """
        prepare resource and pytest code
        """
        tgz = tgz_url[tgz_url.rfind("/") + 1 : -1]

        time_stamp = tgz[0 : tgz.find("^")]

        tmp = tgz.replace(time_stamp + "^", "")
        repo = tmp[0 : tmp.find("^")]

        tmp = tgz.replace(time_stamp + "^" + repo + "^", "")
        model_name = tmp[0 : tmp.find("^")]
        model_path = model_name + "_upload"

        tmp = tgz.replace(time_stamp + "^" + repo + "^" + model_name + "^", "")
        paddle_commit = tmp[0 : tmp.find("^")]

        tmp = tgz.replace(time_stamp + "^" + repo + "^" + model_name + "^" + paddle_commit + "^", "")
        repo_commit = tmp[0 : tmp.find(".")]

        str_all = ""
        for opset_v in self.opset_v_list:
            tmp = (
                "def test_opt_v{}():\n"
                '    """test {} opt version {}"""\n'
                "    logging.info('time stamp: {} !!!')\n"
                "    logging.info('model name: {} !!!')\n"
                "    logging.info('paddle commit: {} !!!')\n"
                "    logging.info('repo commit: {} !!!')\n"
                "    unit_exit_code = os.system(\n"
                '        "paddle2onnx --model_dir={} "\n'
                '        "--model_filename=inference.pdmodel "\n'
                '        "--params_filename=inference.pdiparams "\n'
                '        "--save_file={} "\n'
                '        "--opset_version={} --enable_onnx_checker=True"\n'
                "    )\n"
                "    assert unit_exit_code == 0\n"
                "\n"
                "\n".format(
                    opset_v,
                    model_name,
                    opset_v,
                    time_stamp,
                    model_name,
                    paddle_commit,
                    repo_commit,
                    model_path,
                    os.path.join(model_path, "inference.onnx"),
                    opset_v,
                )
            )
            str_all += tmp

        with open("test_{}.py".format(model_name), "w") as f:
            f.write(
                "#!/bin/env python\n"
                "# -*- coding: utf-8 -*-\n"
                "# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python\n"
                '"""\n'
                "test {} to onnx\n"
                '"""\n'
                "import os\n"
                "import logging\n"
                "\n"
                "\n".format(model_name)
            )
            f.write(str_all)

        os.system("wget -q --no-proxy {}".format(tgz_url))
        os.system("tar -xzf {}".format(tgz))

        return tgz, model_name, model_path

    def run(self):
        """
        run test
        """
        for tgz_url in self.model_url_list:
            tgz, model_name, model_path = self.prepare_resource(tgz_url)

            if platform.system() == "Windows":
                os.system("python.exe -m pytest {} --alluredir=report".format("test_" + model_name + ".py"))
            else:
                os.system("python -m pytest {} --alluredir=report".format("test_" + model_name + ".py"))
            os.remove(tgz)
            shutil.rmtree(model_path)


if __name__ == "__main__":
    test = ClasV2Test()
    test.run()
