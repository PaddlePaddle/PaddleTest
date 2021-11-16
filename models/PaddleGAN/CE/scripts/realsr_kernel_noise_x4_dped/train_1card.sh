export FLAGS_cudnn_deterministic=True
cd ${Project_path}
sed -i 's/epochs/total_iters/g' configs/realsr_kernel_noise_x4_dped.yaml #将epcoh换为iter
sed -i 's/decay_total_iters/decay_epochs/g' configs/realsr_kernel_noise_x4_dped.yaml #恢复学习率衰减字段
sed -i 's/interval:/interval: 99999 #/g' configs/realsr_kernel_noise_x4_dped.yaml #将epcoh换为iter

rm -rf data
ln -s ${Data_path} data
if [ ! -d "log" ]; then
  mkdir log
fi
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install -r requirements.txt
python tools/main.py -c configs/realsr_kernel_noise_x4_dped.yaml -o total_iters=100 log_config.interval=10 > log/realsr_kernel_noise_x4_dped_1card.log 2>&1
cat log/realsr_kernel_noise_x4_dped_1card.log | grep " INFO: Iter: 100/100" > ../log/realsr_kernel_noise_x4_dped_1card.log
cat log/realsr_kernel_noise_x4_dped_1card.log
