home=$PWD
# base
cd paddlebase
rm -rf ./result.txt
echo "[paddlebase cases result]" >> result.txt
bash ./run.sh
paddlebase=$?
echo ${paddlebase}
cat ./result.txt
cd $home


# nn
cd nn
rm -rf ./result.txt
echo "[nn cases result]" >> result.txt
bash ./run.sh
nn=$?
echo ${nn}
cat ./result.txt
cd $home

# optimizer
cd optimizer
rm -rf ./result.txt
echo "[optimizer cases result]" >> result.txt
bash ./run.sh
optimizer=$?
echo ${optimizer}
cat ./result.txt
cd $home


# result
echo "=============== result ================="
if [ `expr ${paddlebase} + ${nn} + ${optimizer}` -eq 0 ]; then
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
