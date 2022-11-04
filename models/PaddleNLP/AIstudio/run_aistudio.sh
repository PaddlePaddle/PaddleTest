#!/usr/bin/env bash
###====install alluer====

# wget -q https://xly-devops.bj.bcebos.com/tools/allure-2.19.0.zip
# unzip allure-2.19.0.zip
# ln -s %{PWD}/allure-2.19.0/bin/allure  /usr/bin/allure

# yum update -y > /dev/null
# yum install curl -y
# yum install -y nodejs > /dev/null
# yum install -y java-1.8.0-openjdk-devel.x86_64 > /dev/null
# export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk
# export JRE_HOME=$JAVA_HOME/jre
# export CLASSPATH=$JAVA_HOME/lib:$JRE_HOME/lib:$CLASSPATH
# export PATH=$JAVA_HOME/bin:$JRE_HOME/bin:$PATH
# export PATH=/usr/bin/allure:$PATH
# which java
# which allure

python -m pytest -sv test_paddlenlp_aistudio.py::test_aistudio_case --alluredir=./result
exit_code=$?
python gen_allure_report.py
exit exit_code
