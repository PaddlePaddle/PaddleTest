cp run_model_mnist.sh models/official/vision/image_classification/ >log.tmp 2>&1
cp is_conv.py models/official/vision/image_classification/ >log.tmp 2>&1
cd models/official/vision/image_classification  >log.tmp 2>&1

sh run_model_mnist.sh
flag=$?
return ${flag}
