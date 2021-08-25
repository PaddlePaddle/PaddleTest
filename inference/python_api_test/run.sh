home=$PWD
cd test_class_model
rm -rf ./result.txt
echo "[Class model inference cases result]" >> result.txt
bash ./run.sh
bug=$?
cd ..
echo "===============Class model result ================="
cat ./test_class_model/result.txt
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
cd ..
echo "===============Detection model result ================="
cat ./test_det_model/result.txt
if [ `expr ${bug}` -eq 0 ]; then
  echo "success!"
else
  echo "error!"
  exit 8
fi

cd test_ocr_model
rm -rf ./result.txt
echo "[Ocr model inference cases result]" >> result.txt
bash ./run.sh
bug=$?
cd ..
echo "===============Ocr model result ================="
cat ./test_ocr_model/result.txt
if [ `expr ${bug}` -eq 0 ]; then
  echo "success!"
else
  echo "error!"
  exit 8
fi
