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
cd ./examples/parallelwave_gan/baker
# train
sed -i "s/train_max_steps: 400000/train_max_steps: 500/g;s/save_interval_steps: 5000/save_interval_steps: 500/g;s/eval_interval_steps: 1000/eval_interval_steps: 500/g;s/batch_size: 8/batch_size: 2/g"  ./conf/default.yaml
rm -rf exp
FLAGS_cudnn_exhaustive_search=true \
FLAGS_conv_workspace_size_limit=4000 \
python train.py \
    --train-metadata=dump/train/norm/metadata.jsonl \
    --dev-metadata=dump/dev/norm/metadata.jsonl \
    --config=conf/default.yaml \
    --output-dir=exp/default \
    --nprocs=2 > ../../../log/parallel_wavegan_baker_2card.log 2>&1
cat ../../../log/parallel_wavegan_baker_2card.log | grep "500/500" | grep "Rank: 0" | awk 'BEGIN{FS=","} {print $5}' > ../../../../log/parallel_wavegan_baker_2card.log
# synthesis
python synthesize.py \
  --config=conf/default.yaml \
  --checkpoint=exp/default/checkpoints/snapshot_iter_500.pdz \
  --test-metadata=./metadata_10.jsonl \
  --output-dir=exp/default/test
