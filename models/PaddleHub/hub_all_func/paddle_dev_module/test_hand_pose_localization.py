"""hand_pose_localization"""
import os
import paddlehub as hub
import paddle

import cv2

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_hand_pose_localization_predict():
    """hand_pose_localization predict"""
    os.system("hub install hand_pose_localization")
    # use_gpu：是否使用GPU进行预测
    model = hub.Module(name="hand_pose_localization", use_gpu=use_gpu)

    # 调用关键点检测API
    result = model.keypoint_detection(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = model.keypoint_detection(paths=['/PATH/TO/IMAGE'])
    print(result)
    os.system("hub uninstall hand_pose_localization")
