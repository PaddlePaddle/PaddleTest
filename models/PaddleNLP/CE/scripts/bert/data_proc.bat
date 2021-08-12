@echo off
cd ../..

if not exist log\bert md log\bert
set logpath=%cd%\log\bert

cd models_repo\examples\language_model\bert\

python create_pretraining_data.py --input_file=data/sample_text.txt --output_file=data/training_data.hdf5 --bert_model=bert-base-uncased --max_seq_length=128 --max_predictions_per_seq=20 --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5
