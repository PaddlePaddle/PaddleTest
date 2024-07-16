python -m scripts.pose2vid --config ./configs/inference/animation.yaml -W 600 -H 784 -L 120


# 检查命令是否成功执行
if [ $? -ne 0 ]; then
  exit 1
fi
