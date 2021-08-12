wget https://paddle-inference-dist.bj.bcebos.com/AI-Rank/models/Paddle/ResNet50.tar.gz
tar -xzvf ResNet50.tar.gz
python3.6 -m paddle_serving_app.package --get_model resnet_v2_50_imagenet
tar -xzvf resnet_v2_50_imagenet.tar.gz
python3.6 encrypt.py
