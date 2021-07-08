home=$PWD
# nn
cd nn
rm -rf ./result.txt
echo "[nn cases result]" >> result.txt
bash ./run.sh
cat ./result.txt
cd $home

# optimizer
cd optimizer
rm -rf ./result.txt
echo "[optimizer cases result]" >> result.txt
bash ./run.sh
cat ./result.txt
cd $home
