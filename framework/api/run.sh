home=$PWD
# nn
cd nn
rm -rf ./result.txt
echo "[nn cases result]" >> result.txt
bash ./run.sh
cat ./result.txt
cd $home
