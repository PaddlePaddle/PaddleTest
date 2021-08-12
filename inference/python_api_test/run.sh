home=$PWD
# base
cd test_class_model
rm -rf ./result.txt
echo "[Class model inference cases result]" >> result.txt
bash ./run.sh
bug=$?
cat ./result.txt
cd ..
echo "===============Class model result ================="
if [ `expr ${bug}` -eq 0 ]; then
  echo "success!"
else
  echo "error!"
  exit 8
fi

cd test_det_model
rm -rf ./result.txt
echo "[Detection model inference cases result]" >> result.txt
bash ./run.sh
bug=$?
cat ./result.txt
cd ..
echo "===============Detection model result ================="
if [ `expr ${bug}` -eq 0 ]; then
  echo "success!"
else
  echo "error!"
  exit 8
fi
