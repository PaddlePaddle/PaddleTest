export FLAGS_cudnn_deterministic=True
echo ${Project_path}
echo ${Data_path}
ls;
pwd;
cd ${Project_path}
pwd;

mkdir log
python -m pip install -e .     
cd ./examples/tacotron2
rm -rf LJSpeech-1.1
ln -s ${Data_path}/train_data/LJSpeech-1.1 ./
# train
rm -rf output
python train.py --data=ljspeech_tacotron2 --output=output --device=gpu --nprocs=2 --opts data.batch_size 2 training.max_iteration 500 training.valid_interval 500 training.save_interval 500 > ../../log/tacotron2_2card.log 2>&1
cat ../../log/tacotron2_2card.log | grep "step: 500" | grep "Rank: 0" | awk 'BEGIN{FS=","} {print $4}' > ../../../log/tacotron2_2card.log
# synthesize
rm -rf exp
python synthesize.py --checkpoint_path=output/checkpoints/step-500 --input=sentences.txt --output=exp/ --device=gpu

