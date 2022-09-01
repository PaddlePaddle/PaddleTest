# rm -rf PaddleOCR/rec_*
# curl -sL https://deb.nodesource.com/setup_8.x | sudo -E bash -
# ubuntu
# apt-get update
# apt-get install -y nodejs
# apt install -y openjdk-8-jdk

# centos
yum update -y > /dev/null
yum install -y nodejs > /dev/null
yum install -y java-1.8.0-openjdk.x86_6 > /dev/null
yum install -y java-1.8.0-openjdk-devel.x86_64 > /dev/null
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.342.b07-1.el7_9.x86_64
export JRE_HOME=$JAVA_HOME/jre
export CLASSPATH=$JAVA_HOME/lib:$JRE_HOME/lib:$CLASSPATH
export PATH=$JAVA_HOME/bin:$JRE_HOME/bin:$PATH
export PATH=/usr/bin/allure:$PATH
rm -rf /usr/bin/allure
ln -s /workspace/AutomaticTestSystem/allure/bin/allure /usr/bin/allure
python -m pip install -r requirements.txt

# export CUDA_VISIBLE_DEVICES=0,1
which allure

python -m pytest -sv $1  --alluredir=./result #--alluredir用于指定存储测试结果的路径)
echo 'exit_code:'$?
cp environment/environment.properties_linux ./result 
mv ./result/environment.properties_linux ./result/environment.properties
allure generate ./result/ -o ./report_test/ --clean
exit $exit_code
# python -m  pytest -sv test_ocr_acc.py --html=rec_report.html --capture=tee-sys
