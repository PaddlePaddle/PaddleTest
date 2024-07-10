# 图像衔接

ppdiffusers_path=PaddleMIX/ppdiffusers
export PYTHONPATH=$ppdiffusers_path:$PYTHONPATH
python scripts/inference-long.py --num-frames 18 --image-size 224 300 --sample-name connect --prompt 'A breathtaking sunrise scene.{"reference_path": "assets/images/condition/sunset1.png;assets/images/condition/sunset2.png","mask_strategy": "0;0,1,0,-1,1"}'


# 检查命令是否成功执行
if [ $? -ne 0 ]; then
  exit 1
fi
