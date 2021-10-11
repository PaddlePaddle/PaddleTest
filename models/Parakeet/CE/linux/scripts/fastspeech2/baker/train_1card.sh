export FLAGS_cudnn_deterministic=True
echo ${Project_path}
echo ${Data_path}
ls;
pwd;
cd ${Project_path}
pwd;

rm -rf ~/datasets
ln -s ${Data_path}/train_data ~/datasets
mkdir log
python -m pip install -e .     
cd ./examples/fastspeech2/baker
# data preprocess
if [ ! -f "baker_alignment_tone.tar.gz" ]; then
  wget https://paddlespeech.bj.bcebos.com/MFA/BZNSYP/with_tone/baker_alignment_tone.tar.gz
  tar xf baker_alignment_tone.tar.gz
fi
sed -i "s/python3/python/g" preprocess.sh
bash preprocess.sh
# train
sed -i "s/max_epoch: 1000/max_epoch: 5/g;s/batch_size: 64/batch_size: 16/g"  ./conf/default.yaml
rm -rf exp
python train.py \
    --train-metadata=dump/train/norm/metadata.jsonl \
    --dev-metadata=dump/dev/norm/metadata.jsonl \
    --config=conf/default.yaml \
    --output-dir=exp/default \
    --nprocs=1 \
    --phones-dict=dump/phone_id_map.txt > ../../../log/fastspeech2_baker_1card.log 2>&1
sed -i "s/max_epoch: 5/max_epoch: 1000/g;s/batch_size: 16/batch_size: 64/g"  ./conf/default.yaml
cat ../../../log/fastspeech2_baker_1card.log | grep "3060/3060" | awk 'BEGIN{FS=","} {print $7}' > ../../../../log/fastspeech2_baker_1card.log
# synthesis
if [ ! -f "pwg_baker_ckpt_0.4.zip" ]; then
  wget https://paddlespeech.bj.bcebos.com/Parakeet/pwg_baker_ckpt_0.4.zip
  unzip pwg_baker_ckpt_0.4.zip
fi
head -10 ./dump/test/norm/metadata.jsonl > ./metadata_10.jsonl
python synthesize.py \
  --fastspeech2-config=conf/default.yaml \
  --fastspeech2-checkpoint=exp/default/checkpoints/snapshot_iter_3060.pdz \
  --fastspeech2-stat=dump/train/speech_stats.npy \
  --pwg-config=pwg_baker_ckpt_0.4/pwg_default.yaml \
  --pwg-checkpoint=pwg_baker_ckpt_0.4/pwg_snapshot_iter_400000.pdz \
  --pwg-stat=pwg_baker_ckpt_0.4/pwg_stats.npy \
  --test-metadata=./metadata_10.jsonl \
  --output-dir=exp/default/test \
  --device="gpu" \
  --phones-dict=dump/phone_id_map.txt



