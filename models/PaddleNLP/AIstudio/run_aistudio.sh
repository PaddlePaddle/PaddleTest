#!/usr/bin/env bash
python -m pip list
ls
mkdir log
python -m pytest -sv test_paddlenlp_aistudio.py::test_aistudio_case --alluredir=./result
# exit_code=$?
python gen_allure_report.py
# exit exit_code
