home=$PWD
echo "branch is ${AGILE_COMPILE_BRANCH}"
python3.7 -m pip install pytest
python3.7 -m pip install scipy


## jit/api
#cd api
#rm -rf ./result.txt
#echo "[jit/api cases result]" >> result.txt
#bash ./run.sh
#jit_api=$?
#echo ${jit_api}
#cat ./result.txt
#cd $home
#
#
## jit/layer
#cd layer
#rm -rf ./result.txt
#echo "[jit/layer cases result]" >> result.txt
#bash ./run.sh
#jit_layer=$?
#echo ${jit_layer}
#cat ./result.txt
#cd $home
#
#
## jit/scene
#cd scene
#rm -rf ./result.txt
#echo "[jit/scene cases result]" >> result.txt
#bash ./run.sh
#jit_scene=$?
#echo ${jit_scene}
#cat ./result.txt
#cd $home
#
## result
#echo "=============== result ================="
#if [ `expr ${jit_api} + ${jit_layer} + ${jit_scene}` -eq 0 ]; then
#  result=`find . -name "result.txt"`
#  for file in ${result}
#    do
#      cat ${file}
#    done
#  echo "success!"
#else
#  result=`find . -name "result.txt"`
#  for file in ${result}
#    do
#      cat ${file}
#    done
#  echo "error!"
#  exit 8
#fi
