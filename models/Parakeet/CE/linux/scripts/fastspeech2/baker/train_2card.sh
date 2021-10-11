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
# train
sed -i "s/max_epoch: 1000/max_epoch: 2/g;s/batch_size: 64/batch_size: 16/g"  ./conf/default.yaml
rm -rf exp
python train.py \
    --train-metadata=dump/train/norm/metadata.jsonl \
    --dev-metadata=dump/dev/norm/metadata.jsonl \
    --config=conf/default.yaml \
    --output-dir=exp/default \
    --nprocs=2 \
    --phones-dict=dump/phone_id_map.txt > ../../../log/fastspeech2_baker_2card.log 2>&1
cat ../../../log/fastspeech2_baker_2card.log | grep "612/612" | grep "Rank: 0" | awk 'BEGIN{FS=","} {print $7}' > ../../../../log/fastspeech2_baker_2card.log
# synthesis
python synthesize.py \
  --fastspeech2-config=conf/default.yaml \
  --fastspeech2-checkpoint=exp/default/checkpoints/snapshot_iter_612.pdz \
  --fastspeech2-stat=dump/train/speech_stats.npy \
  --pwg-config=pwg_baker_ckpt_0.4/pwg_default.yaml \
  --pwg-checkpoint=pwg_baker_ckpt_0.4/pwg_snapshot_iter_400000.pdz \
  --pwg-stat=pwg_baker_ckpt_0.4/pwg_stats.npy \
  --test-metadata=./metadata_10.jsonl \
  --output-dir=exp/default/test \
  --device="gpu" \
  --phones-dict=dump/phone_id_map.txt

