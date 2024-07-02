# Video extending and editting

ppdiffusers_path=PaddleMIX/ppdiffusers
export PYTHONPATH=$ppdiffusers_path:$PYTHONPATH
# video extending
python scripts/inference-long.py --num-frames 12 --image-size 240 240 --sample-name video_extend  --prompt 'A car driving on the ocean.{"reference_path": "./assets/videos/d0_proc.mp4","mask_strategy": "0,0,0,-6,6"}'

# 检查命令是否成功执行
if [ $? -ne 0 ]; then
  exit 1
fi

# video editting
python scripts/inference-long.py --num-frames 16 --image-size 256 256 --sample-name edit --prompt 'A cyberpunk-style car at New York city.{"reference_path": "./assets/videos/d0_proc.mp4","mask_strategy": "0,0,0,0,16,0.4"}'

# 检查命令是否成功执行
if [ $? -ne 0 ]; then
  exit 1
fi
