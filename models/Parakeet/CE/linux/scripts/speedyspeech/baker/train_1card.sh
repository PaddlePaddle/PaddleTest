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
cd ./examples/speedyspeech/baker
# data preprocess
bash preprocess.sh
# train
sed -i "s/max_epoch: 300/max_epoch: 10/g"  ./conf/default.yaml
rm -rf exp
python train.py \
    --train-metadata=dump/train/norm/metadata.jsonl \
    --dev-metadata=dump/dev/norm/metadata.jsonl \
    --config=conf/default.yaml \
    --output-dir=exp/default \
    --nprocs=1 > ../../../log/speedyspeech_baker_1card.log 2>&1
sed -i "s/max_epoch:10/max_epoch: 300/g"  ./conf/default.yaml
cat ../../../log/speedyspeech_baker_1card.log | grep "3060/3060" | awk 'BEGIN{FS=","} {print $6}' > ../../../../log/speedyspeech_baker_1card.log
# synthesize_e2e
if [ ! -f "pwg_baker_ckpt_0.4.zip" ]; then
  wget https://paddlespeech.bj.bcebos.com/Parakeet/pwg_baker_ckpt_0.4.zip
  unzip pwg_baker_ckpt_0.4.zip
fi
python synthesize_e2e.py \
  --speedyspeech-config=conf/default.yaml \
  --speedyspeech-checkpoint=exp/default/checkpoints/snapshot_iter_3060.pdz \
  --speedyspeech-stat=dump/train/stats.npy \
  --pwg-config=../../parallelwave_gan/baker/conf/default.yaml \
  --pwg-checkpoint=pwg_baker_ckpt_0.4/pwg_snapshot_iter_400000.pdz \
  --pwg-stat=pwg_baker_ckpt_0.4/pwg_stats.npy \
  --text=sentences.txt \
  --output-dir=exp/default/e2e \
  --inference-dir=exp/default/inference \
  --device="gpu"
# inference
python inference.py \
  --inference-dir=exp/default/inference \
  --text=sentences.txt \
  --output-dir=exp/default/pd_infer_out
