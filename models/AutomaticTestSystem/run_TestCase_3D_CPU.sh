# rm -rf PaddleOCR/rec_*
# curl -sL https://deb.nodesource.com/setup_8.x | sudo -E bash -
# ubuntu
# apt-get update
# apt-get install -y nodejs
# apt install -y openjdk-8-jdk

# centos
yum update -y
yum install -y nodejs
yum install -y java-1.8.0-openjdk.x86_6
yum install -y java-1.8.0-openjdk-devel.x86_64
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.342.b07-1.el7_9.x86_64
export JRE_HOME=$JAVA_HOME/jre
export CLASSPATH=$JAVA_HOME/lib:$JRE_HOME/lib:$CLASSPATH
export PATH=$JAVA_HOME/bin:$JRE_HOME/bin:$PATH
export PATH=/usr/bin/allure:$PATH
rm -rf /usr/bin/allure
# ln -s /workspace/AutomaticTestSystem/allure/bin/allure /usr/bin/allure
ln -s /ssd1/jiaxiao01/no_one_ocr/function/AutomaticTestSystem/allure/bin/allure /usr/bin/allure
python -m pip install -r requirements.txt

# export CUDA_VISIBLE_DEVICES=0,1
which allure
rm -rf result
rm -rf report_3D

python -m pytest -sv test_3D_acc.py  --alluredir=./result #--alluredir用于指定存储测试结果的路径)
cp environment/environment.properties_linux ./result_CPU 
mv ./result/environment.properties_linux ./result/environment.properties
unset GREP_OPTIONS
python -c 'import paddle;print("paddle_version={}".format(paddle.__version__))' >> ./result/environment.properties
python -c 'import paddle;print("paddle_commit={}".format(paddle.version.commit))' >> ./result/environment.properties

cd Paddle3D
commit=`git rev-parse HEAD`
cd ..
echo 'Paddle3D_commit='$commit >> ./result/environment.properties
allure generate ./result_CPU/ -o ./report_3D_CPU/ --clean
# python -m  pytest -sv test_ocr_acc.py --html=rec_report.html --capture=tee-sys
