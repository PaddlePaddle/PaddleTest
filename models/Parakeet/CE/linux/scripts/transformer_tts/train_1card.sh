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
cd ./examples/transformer_tts/ljspeech
# data preprocess
sed -i "s/python3/python/g" preprocess.sh
bash preprocess.sh
# train
sed -i "s/max_epoch: 500/max_epoch: 1/g;s/batch_size: 16/batch_size: 2/g"  ./conf/default.yaml
rm -rf exp
python train.py \
    --train-metadata=dump/train/norm/metadata.jsonl \
    --dev-metadata=dump/dev/norm/metadata.jsonl \
    --config=conf/default.yaml \
    --output-dir=exp/default \
    --nprocs=1 \
    --phones-dict=dump/phone_id_map.txt > ../../../log/transformer_tts_1card.log 2>&1
cat ../../../log/transformer_tts_1card.log | grep "6450/6450" | awk 'BEGIN{FS=","} {print $9}' > ../../../../log/transformer_tts_1card.log
sed -i "s/max_epoch: 1/max_epoch: 500/g;s/batch_size: 2/batch_size: 16/g"  ./conf/default.yaml
# synthesize
if [ ! -f "waveflow_ljspeech_ckpt_0.3.zip" ]; then
  wget https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_ljspeech_ckpt_0.3.zip
  unzip waveflow_ljspeech_ckpt_0.3.zip
fi
head -3 ./dump/test/norm/metadata.jsonl > ./metadata_3.jsonl
python synthesize.py \
  --transformer-tts-config=conf/default.yaml \
  --transformer-tts-checkpoint=exp/default/checkpoints/snapshot_iter_6450.pdz \
  --transformer-tts-stat=dump/train/speech_stats.npy \
  --waveflow-config=waveflow_ljspeech_ckpt_0.3/config.yaml \
  --waveflow-checkpoint=waveflow_ljspeech_ckpt_0.3/step-2000000.pdparams \
  --test-metadata=./metadata_3.jsonl \
  --output-dir=exp/default/test \
  --device="gpu" \
  --phones-dict=dump/phone_id_map.txt
# synthesize_e2e
head -3 ../sentences.txt > ./sentences_3.txt
python synthesize_e2e.py \
  --transformer-tts-config=conf/default.yaml \
  --transformer-tts-checkpoint=exp/default/checkpoints/snapshot_iter_6450.pdz \
  --transformer-tts-stat=dump/train/speech_stats.npy \
  --waveflow-config=waveflow_ljspeech_ckpt_0.3/config.yaml \
  --waveflow-checkpoint=waveflow_ljspeech_ckpt_0.3/step-2000000.pdparams \
  --text=./sentences_3.txt \
  --output-dir=exp/default/test_e2e \
  --device="gpu" \
  --phones-dict=dump/phone_id_map.txt
