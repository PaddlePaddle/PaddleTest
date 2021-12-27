home=$PWD
echo "branch is ${AGILE_COMPILE_BRANCH}"
python3.7 -m pip install pytest
python3.7 -m pip install scipy


# api
cd api
#rm -rf ./result.txt
#echo "[api cases result]" >> result.txt
bash ./run.sh
api=$?
echo ${api}
#cat ./result.txt
cd $home


# jit
cd e2e
cd jit
#rm -rf ./result.txt
#echo "[nn cases result]" >> result.txt
bash ./run.sh
jit=$?
echo ${jit}
#cat ./result.txt
cd $home


# result
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "=============== final-result ================="
if [ `expr ${api} + ${jit}` -eq 0 ]; then
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
