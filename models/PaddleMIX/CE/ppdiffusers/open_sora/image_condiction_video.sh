# 以图像为条件，生成视频

ppdiffusers_path=PaddleMIX/ppdiffusers
export PYTHONPATH=$ppdiffusers_path:$PYTHONPATH
python scripts/inference-long.py --num-frames 20 --image-size 224 300 --sample-name image-cond --prompt 'A breathtaking sunrise scene.{"reference_path": "assets/images/condition/wave.png","mask_strategy": "0"}'

# 检查命令是否成功执行
if [ $? -ne 0 ]; then
  exit 1
fi
