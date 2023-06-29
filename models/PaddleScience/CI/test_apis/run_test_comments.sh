error_num=0
python${py_version} test_comments.py
if [ $? != 0 ];then
  echo "test_commments failed! please check your code"
  error_num=$((error_num+1))

fi
exit ${error_num}