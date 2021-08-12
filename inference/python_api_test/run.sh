home=$PWD
# base
cd test_class_model
rm -rf ./result.txt
echo "[Class model inference cases result]" >> result.txt
bash ./run.sh
class_model=$?
echo ${class_model}
cat ./result.txt
cd ..
# result
echo "=============== result ================="
if [ `expr ${class_model}` -eq 0 ]; then
  result=`find . -name "result.txt"`
  for file in ${result}
    do
      cat ${file}
    done
  echo "success!"
else
  result=`find . -name "result.txt"`
  for file in ${result}
    do
      cat ${file}
    done
  echo "error!"
  exit 8
fi



cd test_det_model
rm -rf ./result.txt
echo "[Detection model inference cases result]" >> result.txt
bash ./run.sh
class_model=$?
echo ${class_model}
cat ./result.txt
# result
echo "=============== result ================="
if [ `expr ${class_model}` -eq 0 ]; then
  result=`find . -name "result.txt"`
  for file in ${result}
    do
      cat ${file}
    done
  echo "success!"
else
  result=`find . -name "result.txt"`
  for file in ${result}
    do
      cat ${file}
    done
  echo "error!"
  exit 8
fi
