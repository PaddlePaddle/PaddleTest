# 前向推理

ppdiffusers_path=PaddleMIX/ppdiffusers
export PYTHONPATH=$ppdiffusers_path:$PYTHONPATH
python scripts/inference.py --prompt "A beautiful sunset over the city" --num-frames 16 --image-size 256 256

# 检查命令是否成功执行
if [ $? -ne 0 ]; then
  exit 1
fi
