export FLAGS_cudnn_deterministic=True
echo ${Project_path}
echo ${Data_path}
ls;
pwd;
cd ${Project_path}
pwd;

mkdir log
python -m pip install -e .
cd ./examples/waveflow
rm -rf LJSpeech-1.1
ln -s ${Data_path}/train_data/LJSpeech-1.1 ./
# train
rm -rf runs
python train.py --data=ljspeech_waveflow/ --output=runs/test --device="gpu" --nprocs=2 --opts data.batch_size 2 training.max_iteration 500 training.valid_interval 500 training.save_interval 500 > ../../log/waveflow_2card.log 2>&1
cat ../../log/waveflow_2card.log | grep "step: 500" | grep "Rank: 0" | awk 'BEGIN{FS=","} {print $4}' > ./tmp_2card.log
sed -i "s/-//g" ./tmp_2card.log
cat tmp_2card.log > ../../../log/waveflow_2card.log
# synthesize
ln -s ${Data_path}/preprocess_data/mels ./
rm -rf wavs
python synthesize.py --input=mels/ --output=wavs/ --checkpoint_path=runs/test/checkpoints/step-500 --device="gpu" --verbose
