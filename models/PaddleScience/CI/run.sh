# 获取当前工作目录
home=$PWD

# 运行 api 测试脚本，并获取退出码
cd test_apis
bash ./run.sh
api=$?
echo ${api}
cd $home

# 运行 examples 测试脚本，并获取退出码
cd test_models
bash ./run.sh
example=$?
echo ${example}
cd $home
# 输出测试结果
echo "=============== result ================="
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
if [ `expr ${api} + ${example}` -eq 0 ]; then
  # 如果两个测试脚本的退出码之和为 0，则说明测试全部通过，输出正确结果并退出程序
  result=`find . -name "result.txt"`
  for file in ${result}
    do
      cat ${file}
      echo "------------------------"
    done
  echo "success!"
else
  # 如果两个测试脚本的退出码之和不为 0，则说明有测试未通过，输出错误结果并退出程序
  result=`find $home/ -name "result.txt"`
  for file in ${result}
    do
      cat ${file}
      echo "--------------------------"
    done
  echo "error!"
fi
echo `expr ${api} + ${example}`
exit `expr ${api} + ${example}`

