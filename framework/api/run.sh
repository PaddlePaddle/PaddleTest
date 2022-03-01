home=$PWD
echo "branch is ${AGILE_COMPILE_BRANCH}"
python3.7 -m pip install pytest
python3.7 -m pip install scipy


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

# loss
cd loss
rm -rf ./result.txt
echo "[loss cases result]" >> result.txt
bash ./run.sh
loss=$?
echo ${loss}
cat ./result.txt
cd $home

# device
cd device
rm -rf ./result.txt
echo "[device cases result]" >> result.txt
bash ./run.sh
device=$?
echo ${device}
cat ./result.txt
cd $home

# incubate
cd incubate
rm -rf ./result.txt
echo "[incubate cases result]" >> result.txt
bash ./run.sh
incubate=$?
echo ${incubate}
cat ./result.txt
cd $home

# linalg
cd linalg
rm -rf ./result.txt
echo "[linalg cases result]" >> result.txt
bash ./run.sh
linalg=$?
echo ${linalg}
cat ./result.txt
cd $home

# fft
cd fft
rm -rf ./result.txt
echo "[fft cases result]" >> result.txt
bash ./run.sh
fft=$?
echo ${fft}
cat ./result.txt
cd $home

# utils
cd utils
rm -rf ./result.txt
echo "[utils cases result]" >> result.txt
bash ./run.sh
utils=$?
echo ${utils}
cat ./result.txt
cd $home

# distribution
cd distribution
rm -rf ./result.txt
echo "[distribution cases result]" >> result.txt
bash ./run.sh
distribution=$?
echo ${distribution}
cat ./result.txt
cd $home

# result
echo "=============== result ================="
if [ `expr ${paddlebase} + ${nn} + ${optimizer} + ${loss} + ${device} + ${incubate} + ${linalg} + ${ffti} +${utils} +${distribution}` -eq 0 ]; then
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
