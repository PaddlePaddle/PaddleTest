home=$PWD
# base
cd test_class_model
rm -rf ./result.txt
echo "[class model inference cases result]" >> result.txt
bash ./run.sh
cat ./result.txt
cd ..

cd test_det_model
rm -rf ./result.txt
echo "[class model inference cases result]" >> result.txt
bash ./run.sh
cat ./result.txt
